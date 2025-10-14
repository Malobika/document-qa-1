# app.py
# -*- coding: utf-8 -*-

import os
import uuid
import re
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import streamlit as st

from openai import OpenAI

# --- ChromaDB (persistent local vector store)
__import__("pysqlite3")  # shim for some hosts where sqlite3 is missing features
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb

# NEW: HTML fetching/parsing
import requests
from bs4 import BeautifulSoup


# -----------------------------
# Config
# -----------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "news_chunks"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # adjust if needed

# Max characters to index per doc (roughly ~8k tokens safety margin)
MAX_CHARS = 40_000
REQ_TIMEOUT = 15  # seconds
USER_AGENT = (
    "Mozilla/5.0 (compatible; CSVNewsBot/1.0; +https://example.local) "
    "PythonRequests/2.x"
)

# “Interestingness” for law-firm context
LAW_KEYWORDS = [
    "litigation", "lawsuit", "regulation", "regulatory", "antitrust", "merger",
    "acquisition", "m&a", "sanctions", "compliance", "data privacy", "gdpr",
    "ccpa", "patent", "ip", "governance", "enforcement", "settlement", "class action"
]


# -----------------------------
# Helpers
# -----------------------------
def validate_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    client = OpenAI(api_key=api_key)
    _ = client.models.list()
    return client


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or " ").strip()
    return s[:MAX_CHARS]


def html_to_text(url: str) -> Optional[str]:
    """
    Fetch a URL and extract main text content from HTML.
    Returns None on failure, or cleaned text on success.
    """
    if not url or not isinstance(url, str):
        return None
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"},
            timeout=REQ_TIMEOUT,
        )
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None

    ctype = resp.headers.get("Content-Type", "").lower()
    if "text/html" not in ctype:
        # Not HTML (pdf, json, etc.) → skip
        return None

    try:
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style/nav/footer headers—typical boilerplate
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
            tag.decompose()

        # Prefer article/main if present, else body
        container = soup.find("article") or soup.find("main") or soup.body or soup
        text = container.get_text(separator=" ", strip=True)

        # Optional: include title and meta description up top
        title = soup.title.get_text(strip=True) if soup.title else ""
        desc = ""
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            desc = md["content"].strip()

        combined = "\n".join([x for x in [title, desc, text] if x])
        return clean_text(combined)
    except Exception:
        return None


def load_csv(filepath_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(filepath_or_buffer)

    # Normalize expected columns if present
    cols = {c.lower(): c for c in df.columns}

    # Ensure URL column (case-insensitive) → standardize as 'URL'
    if "url" in cols and "URL" not in df.columns:
        df["URL"] = df[cols["url"]]
    elif "URL" not in df.columns:
        # No URL column at all; still OK (we’ll just index CSV text)
        df["URL"] = None

    # Date → 'Date'
    if "date" in cols and "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df[cols["date"]], errors="coerce")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.NaT

    # Company name (optional)
    if "company_name" in cols and "company_name" not in df.columns:
        df["company_name"] = df[cols["company_name"]]
    elif "company_name" not in df.columns:
        df["company_name"] = None

    return df


def enrich_from_urls(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row with a URL, fetch HTML and extract text into 'FetchedText'.
    Also build/overwrite 'Document' with the fetched text when available.
    If fetch fails, we later fall back to concatenating string columns.
    """
    if "FetchedText" not in df.columns:
        df["FetchedText"] = None

    urls = df["URL"].fillna("").astype(str).tolist()
    total = len(urls)
    if total == 0:
        return df

    st.info("Fetching article text from URLs…")
    prog = st.progress(0.0)
    fetched_count = 0

    for i, u in enumerate(urls):
        text = html_to_text(u) if u else None
        if text:
            df.at[i, "FetchedText"] = text
            fetched_count += 1
        prog.progress((i + 1) / total)

    st.success(f"Fetched {fetched_count}/{total} pages.")

    # Build Document column:
    if "Document" not in df.columns:
        df["Document"] = None

    # If fetched text exists, use it; else we will fill Document from CSV text later
    for i in range(len(df)):
        if pd.notna(df.at[i, "FetchedText"]) and str(df.at[i, "FetchedText"]).strip():
            df.at[i, "Document"] = df.at[i, "FetchedText"]

    # For rows still missing Document, fallback: concatenate other string columns
    missing = df["Document"].isna() | (df["Document"].astype(str).str.strip() == "")
    if missing.any():
        str_cols = [c for c in df.columns if df[c].dtype == "object" and c not in ("Document", "FetchedText")]
        if str_cols:
            fallback_text = df.loc[missing, str_cols].astype(str).agg(" | ".join, axis=1).map(clean_text)
            df.loc[missing, "Document"] = fallback_text
        else:
            # If nothing to concatenate, set empty strings to avoid None
            df.loc[missing, "Document"] = ""

    return df


def chunk_rows(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Each row becomes one chunk: Document text + metadata.
    """
    documents, metadatas, ids = [], [], []
    for idx, row in df.iterrows():
        text = str(row.get("Document", "") or "")
        text = clean_text(text)
        if not text.strip():
            continue
        meta = {
            "row_index": int(idx),
            "company_name": str(row.get("company_name", "")) if pd.notna(row.get("company_name", "")) else "",
            "date": str(row.get("Date", "")) if pd.notna(row.get("Date", "")) else "",
            "url": str(row.get("URL", "")) if pd.notna(row.get("URL", "")) else "",
            "source": "html" if pd.notna(row.get("FetchedText", None)) and str(row.get("FetchedText")).strip() else "csv",
        }
        doc_id = str(uuid.uuid4())
        documents.append(text)
        metadatas.append(meta)
        ids.append(doc_id)
    return documents, metadatas, ids


def build_vector_db(docs: List[str], metas: List[Dict[str, Any]], ids: List[str], client: OpenAI):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() > 0:
        return collection

    embeddings = []
    batch_size = 128
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([d.embedding for d in resp.data])

    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return collection


def query_collection(collection, client: OpenAI, query: str, k: int = 10) -> Dict[str, Any]:
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    result = collection.query(
        query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"]
    )
    return result


def score_interesting(meta: Dict[str, Any], doc: str) -> float:
    score = 0.0
    low = (doc or "").lower()
    score += sum(1 for k in LAW_KEYWORDS if k in low)
    if meta.get("date"):
        try:
            d = pd.to_datetime(meta["date"], errors="coerce")
            if pd.notna(d):
                days_ago = max(0, (pd.Timestamp.utcnow() - d.tz_localize(None)).days)
                recency = max(0.0, 1.0 - min(days_ago, 365) / 365.0)
                score += 0.5 * recency
        except Exception:
            pass
    # Give a tiny bump if source is HTML (tends to be richer than CSV-only)
    if meta.get("source") == "html":
        score += 0.1
    return score


def rank_most_interesting(results: Dict[str, Any]) -> List[Tuple[float, Dict[str, Any], str]]:
    items = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        items.append((score_interesting(meta, doc), meta, doc))
    items.sort(key=lambda x: x[0], reverse=True)
    return items


    # -----------------------------
    # Streamlit App
    # -----------------------------
def page():
    st.set_page_config(page_title="News Bot (CSV + HTML + Chroma)", page_icon="📰", layout="wide")
    st.title("📰 CSV News Bot (with HTML fetching)")

    with st.sidebar:
        st.markdown("### Setup")
        csv_file = st.file_uploader("Upload your news CSV", type=["csv"])
        keep_msgs = st.slider("Messages to keep", min_value=1, max_value=5, value=3)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages = st.session_state.messages[-keep_msgs:]

    if csv_file is not None:
        # Build client
        try:
            client = validate_openai_client()
        except Exception as e:
            st.error(f"OpenAI client error: {e}")
            st.stop()

        # Load + fetch
        try:
            df = load_csv(csv_file)
            df = enrich_from_urls(df)
            docs, metas, ids = chunk_rows(df)
            if not docs:
                st.warning("No indexable text found (even after fetching URLs).")
            collection = build_vector_db(docs, metas, ids, client)
            st.success("Vector DB ready ✅")
        except Exception as e:
            st.error(f"Data/Vector error: {e}")
            st.stop()

        # Show prior messages
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        user_prompt = st.chat_input("Ask about the news…")
        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            lowered = user_prompt.strip().lower()

            if "find the most interesting news" in lowered:
                res = query_collection(
                    collection,
                    client,
                    query="legal risk regulation litigation merger acquisition class action enforcement compliance",
                    k=25,
                )
                ranked = rank_most_interesting(res)[:10]
                lines = []
                for i, (score, meta, doc) in enumerate(ranked, start=1):
                    title = (doc[:160] + "…") if len(doc) > 160 else doc
                    url = meta.get("url") or ""
                    date_str = meta.get("date") or ""
                    src = meta.get("source") or "csv"
                    lines.append(
                        f"{i}. **Score {score:.2f}** — {date_str} — [{title}]({url})  \n_source: {src}_"
                        if url else f"{i}. **Score {score:.2f}** — {date_str} — {title}  \n_source: {src}_"
                    )
                answer = "Top interesting items (law-firm context):\n\n" + "\n\n".join(lines)

            elif lowered.startswith("find news about "):
                topic = user_prompt[len("find news about "):].strip() or "general"
                res = query_collection(collection, client, query=topic, k=12)
                docs_ = res.get("documents", [[]])[0]
                metas_ = res.get("metadatas", [[]])[0]
                items = []
                for doc, meta in zip(docs_, metas_):
                    date_str = meta.get("date") or ""
                    url = meta.get("url") or ""
                    src = meta.get("source") or "csv"
                    title = (doc[:200] + "…") if len(doc) > 200 else doc
                    items.append(
                        f"- {date_str} — [{title}]({url})  \n_source: {src}_"
                        if url else f"- {date_str} — {title}  \n_source: {src}_"
                    )
                answer = f"News about **{topic}**:\n\n" + ("\n\n".join(items) if items else "No matches found.")

            else:
                # Generic RAG
                res = query_collection(collection, client, query=user_prompt, k=8)
                ctx_docs = res.get("documents", [[]])[0]
                ctx_metas = res.get("metadatas", [[]])[0]
                context_snips = []
                for d, m in zip(ctx_docs, ctx_metas):
                    url = m.get("url") or ""
                    date_str = m.get("date") or ""
                    snippet = d if len(d) < 700 else d[:700] + "…"
                    line = f"[{date_str}] {snippet}"
                    if url:
                        line += f"\nURL: {url}"
                    context_snips.append(line)
                context_text = "\n\n---\n\n".join(context_snips) if context_snips else "No context found."

                messages = [
                    {"role": "system", "content": "You are a news assistant for a large global law firm. Be concise, risk-aware, and practical."},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": f"Context (retrieved):\n\n{context_text}"},
                ]
                resp = client.chat.completions.create(model=CHAT_MODEL, temperature=0.2, messages=messages)
                answer = resp.choices[0].message.content

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

    else:
        st.info("Upload a CSV to begin.")


def run():
    page()
