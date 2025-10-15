# app.py
# minimal CSV -> HTML fetch -> Chroma -> Streamlit chat

import os, uuid, re, sys
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# sqlite shim for some hosts (Chroma needs it)
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import requests
from bs4 import BeautifulSoup

# ---------- config ----------
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "news_chunks"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"   # adjust if needed
MAX_CHARS = 40_000
REQ_TIMEOUT = 15
USER_AGENT = "CSVNewsBot/1.0 (+local) PythonRequests"

LAW_KEYWORDS = [
    "litigation", "lawsuit", "regulation", "regulatory", "antitrust", "merger",
    "acquisition", "m&a", "sanctions", "compliance", "data privacy", "gdpr",
    "ccpa", "patent", "ip", "governance", "enforcement", "settlement", "class action"
]

# ---------- helpers ----------
def openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in env.")
    cli = OpenAI(api_key=key)
    _ = cli.models.list()
    return cli

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s[:MAX_CHARS]

def html_to_text(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"},
            timeout=REQ_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        if "text/html" not in r.headers.get("Content-Type", "").lower():
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
            t.decompose()
        container = soup.find("article") or soup.find("main") or soup.body or soup
        text = container.get_text(" ", strip=True)
        title = soup.title.get_text(strip=True) if soup.title else ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        desc = meta_desc.get("content", "").strip() if meta_desc else ""
        return clean_text("\n".join([x for x in (title, desc, text) if x]))
    except Exception:
        return None

def load_csv_exact(file) -> pd.DataFrame:
    """Assumes columns: company_name, days_since_2000, Date, Document, URL"""
    df = pd.read_csv(file)
    # enforce exact names (case-insensitive remap if needed)
    lower = {c.lower(): c for c in df.columns}
    must = ["company_name", "days_since_2000", "date", "document", "url"]
    missing = [m for m in must if m not in lower]
    if missing:
        raise ValueError(f"CSV must have columns: {must}. Missing: {missing}")
    # normalize
    df = df.rename(columns={lower["company_name"]: "company_name",
                            lower["days_since_2000"]: "days_since_2000",
                            lower["date"]: "Date",
                            lower["document"]: "Document",
                            lower["url"]: "URL"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df[["company_name", "days_since_2000", "Date", "Document", "URL"]]

def enrich_with_html(df: pd.DataFrame) -> pd.DataFrame:
    st.info("Fetching article text from URL column…")
    prog = st.progress(0.0)
    got = 0
    for i, url in enumerate(df["URL"].astype(str).fillna("")):
        text = html_to_text(url)
        if text:  # prefer fetched text over CSV's Document
            df.at[i, "Document"] = text
            got += 1
        prog.progress((i + 1) / len(df))
    st.success(f"Fetched {got}/{len(df)} pages.")
    # final cleanup
    df["Document"] = df["Document"].astype(str).map(clean_text)
    return df

def to_chroma(df: pd.DataFrame, client: OpenAI):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = chroma_client.get_or_create_collection(COLLECTION_NAME)
    if col.count() > 0:
        return col

    docs, metas, ids = [], [], []
    for i, row in df.iterrows():
        text = (row["Document"] or "").strip()
        if not text:
            continue
        docs.append(text)
        metas.append({
            "company_name": str(row["company_name"] or ""),
            "date": str(row["Date"] or ""),
            "url": str(row["URL"] or ""),
            "days_since_2000": int(row["days_since_2000"]) if pd.notna(row["days_since_2000"]) else None,
        })
        ids.append(str(uuid.uuid4()))

    # embed in small batches
    embs = []
    bs = 128
    for j in range(0, len(docs), bs):
        resp = client.embeddings.create(model=EMBED_MODEL, input=docs[j:j+bs])
        embs.extend([d.embedding for d in resp.data])

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return col

def retrieve(col, client: OpenAI, query: str, k: int = 10):
    q = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    return col.query(query_embeddings=[q], n_results=k, include=["documents","metadatas","distances"])

def score_interesting(meta: Dict[str, Any], doc: str) -> float:
    s = 0.0
    low = (doc or "").lower()
    s += sum(1 for k in LAW_KEYWORDS if k in low)
    # recency bump
    try:
        d = pd.to_datetime(meta.get("date"), errors="coerce")
        if pd.notna(d):
            days = max(0, (pd.Timestamp.utcnow() - d.tz_localize(None)).days)
            s += 0.5 * max(0.0, 1.0 - min(days, 365)/365.0)
    except Exception:
        pass
    return s

def rank_interesting(results) -> List[str]:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    items = []
    for doc, meta in zip(docs, metas):
        items.append((score_interesting(meta, doc), meta, doc))
    items.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for i, (score, meta, doc) in enumerate(items, 1):
        title = (doc[:180] + "…") if len(doc) > 180 else doc
        url = meta.get("url") or ""
        date = meta.get("date") or ""
        line = f"{i}. **{score:.2f}** — {date} — [{title}]({url})" if url else f"{i}. **{score:.2f}** — {date} — {title}"
        lines.append(line)
    return lines[:10]
def page():
    # ---------- UI ----------
    st.set_page_config(page_title="Simple CSV News Bot", page_icon="📰", layout="wide")
    st.title("📰 Simple CSV News Bot")

    with st.sidebar:
        csv_file = st.file_uploader("Upload CSV with columns: company_name, days_since_2000, Date, Document, URL", type=["csv"])
        keep = st.slider("Messages to keep", 1, 5, 3)

    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    st.session_state.msgs = st.session_state.msgs[-keep:]

    if csv_file is None:
        st.info("Upload your CSV to start.")
        st.stop()

    try:
        cli = openai_client()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        st.stop()

    try:
        df = load_csv_exact(csv_file)
        df = enrich_with_html(df)
        col = to_chroma(df, cli)
        st.success("Index ready ✅")
    except Exception as e:
        st.error(f"Data/Index error: {e}")
        st.stop()

    # replay
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Ask: 'find the most interesting news' or 'find news about x' …")
    if prompt:
        st.session_state.msgs.append({"role": "user", "content": prompt})
        low = prompt.strip().lower()

        if "find the most interesting news" in low:
            res = retrieve(col, cli, "litigation regulation enforcement merger acquisition class action compliance", k=25)
            lines = rank_interesting(res)
            ans = "Top interesting items (law-firm context):\n\n" + ("\n\n".join(lines) if lines else "No matches.")

        elif low.startswith("find news about "):
            topic = prompt[len("find news about "):].strip() or "general"
            res = retrieve(col, cli, topic, k=12)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            items = []
            for d, m in zip(docs, metas):
                title = (d[:200] + "…") if len(d) > 200 else d
                date = m.get("date") or ""
                url = m.get("url") or ""
                items.append(f"- {date} — [{title}]({url})" if url else f"- {date} — {title}")
            ans = f"News about **{topic}**:\n\n" + ("\n\n".join(items) if items else "No matches.")

        else:
            # generic RAG summary
            res = retrieve(col, cli, prompt, k=8)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            ctx = []
            for d, m in zip(docs, metas):
                snip = d if len(d) < 700 else d[:700] + "…"
                date = m.get("date") or ""
                url = m.get("url") or ""
                ctx.append(f"[{date}] {snip}\n{('URL: ' + url) if url else ''}".strip())
            context_text = "\n\n---\n\n".join(ctx) if ctx else "No context."

            msgs = [
                {"role": "system", "content": "You are a news assistant for a large global law firm. Be concise and risk-aware."},
                {"role": "user", "content": prompt},
                {"role": "system", "content": f"Context:\n\n{context_text}"},
            ]
            resp = cli.chat.completions.create(model=CHAT_MODEL, temperature=0.2, messages=msgs)
            ans = resp.choices[0].message.content

        st.session_state.msgs.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.write(ans)


def run():
    page()

if __name__ == "__main__":
    run()

