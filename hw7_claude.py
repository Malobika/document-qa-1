# === CLEANED MAIN APP (single file) with Cheap/Expensive Model Switch for Claude & OpenAI ===
import os
import csv
import re
import hashlib
import time
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from anthropic import Anthropic
from openai import OpenAI
import chromadb

# ---- PAGE CONFIG (call exactly once, before any other st.* calls) ----
st.set_page_config(page_title="News Bot (OpenAI/Claude)", page_icon="📰", layout="wide")

# ========== CONFIG ==========
CSV_PATH = "./files/Examples.csv"
CHROMA_PATH = "./chroma_db_claude"
COLLECTION_NAME = "news_claude"
EMBED_MODEL = "text-embedding-3-small"  # Using OpenAI embeddings

# Claude defaults
CLAUDE_MODEL_EXPENSIVE = "claude-3-5-sonnet-20241022"  # keep as-is per your request
CLAUDE_MODEL_CHEAP     = "claude-3-haiku-20240307"

# OpenAI chat defaults
OPENAI_MODEL_EXPENSIVE = "gpt-4o"
OPENAI_MODEL_CHEAP     = "gpt-4o-mini"

LAW_KEYWORDS = [
    "lawsuit", "litigation", "regulation", "antitrust", "merger",
    "acquisition", "compliance", "patent", "settlement"
]

# ========== HELPERS ==========

def get_anthropic_client():
    """Initialize Anthropic client (uses CLAUDE_API_KEY)."""
    api_key = os.getenv("CLAUDE_API_KEY")  # do not change per user request
    if not api_key:
        st.error("Set CLAUDE_API_KEY environment variable")
        st.stop()
    return Anthropic(api_key=api_key)

def get_openai_client():
    """Initialize OpenAI client (embeddings + chat)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Set OPENAI_API_KEY environment variable (needed for embeddings/chat)")
        st.stop()
    return OpenAI(api_key=api_key)

def fetch_url_text(url: str, timeout: int = 12, max_chars: int = 10000) -> str | None:
    """Fetch a URL and return plain text (up to max_chars). Return None on error."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (news-bot/1.0; +https://example.com)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return (text[:max_chars] if len(text) > max_chars else text) or None
    except Exception:
        return None

def load_and_enrich_csv(csv_file: str, fetched_col: str = "fetched_data"):
    """Optional: fetch and persist article text into the CSV."""
    if not os.path.exists(csv_file):
        st.error(f"CSV not found at {csv_file}")
        st.stop()

    df = pd.read_csv(csv_file)
    if fetched_col not in df.columns:
        df[fetched_col] = pd.NA
    if "fetched_at" not in df.columns:
        df["fetched_at"] = pd.NA

    total = len(df)
    st.info(f"Loading {total} articles...")
    progress_bar = st.progress(0)

    updated_rows = 0
    for idx, row in df.iterrows():
        url = row.get("URL")
        existing = row.get(fetched_col)
        if pd.notna(url) and (pd.isna(existing) or (isinstance(existing, str) and existing.strip() == "")):
            fetched_text = fetch_url_text(str(url))
            if fetched_text:
                df.at[idx, fetched_col] = fetched_text
                df.at[idx, "fetched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                updated_rows += 1
        progress_bar.progress((idx + 1) / max(1, total))

    if updated_rows > 0:
        df.to_csv(csv_file, index=False, encoding="utf-8")
        st.success(f"✅ Fetched content for {updated_rows} rows and saved to {csv_file}")
    else:
        st.info("No updates needed — all rows already have fetched data.")
    return df

def load_csv_to_dict():
    articles = []
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                articles.append(row)
    except FileNotFoundError:
        st.error(f"CSV not found at {CSV_PATH}")
        st.stop()
    return articles

def enrich_articles(articles):
    st.info(f"Loading {len(articles)} articles...")
    progress_bar = st.progress(0)
    for i, article in enumerate(articles):
        url = article.get("URL", "")
        if url:
            fetched_text = fetch_url_text(url)
            if fetched_text:
                article["Document"] = fetched_text
        progress_bar.progress((i + 1) / max(1, len(articles)))
    return articles

def _stable_id(url: str, date: str, idx: int) -> str:
    """Deterministic, collision-resistant ID for article chunks."""
    h = hashlib.sha1(f"{url}|{date}|{idx}".encode("utf-8")).hexdigest()
    return f"csv::{h}"

def create_vector_db(articles, openai_client):
    """Create ChromaDB collection with OpenAI embeddings"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    try:
        existing_count = collection.count()
    except Exception:
        existing_count = 0

    if existing_count > 0:
        st.info(f"✅ Loaded existing embeddings ({existing_count} documents)")
        return collection

    st.info("Creating embeddings with OpenAI (this only happens once)...")
    documents, metadatas, ids = [], [], []
    for i, article in enumerate(articles):
        doc_text = article.get("Document", "")
        if len(doc_text) > 50:
            documents.append(doc_text)
            metadatas.append({
                "company": article.get("company_name", ""),
                "date": article.get("Date", ""),
                "url": article.get("URL", "")
            })
            ids.append(f"doc_{i}")

    batch_size = 100
    progress_bar = st.progress(0)
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        response = openai_client.embeddings.create(model=EMBED_MODEL, input=batch_docs)
        embeddings = [item.embedding for item in response.data]

        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
            ids=batch_ids
        )
        progress_bar.progress(min((i + batch_size) / len(documents), 1.0))

    st.success(f"✅ Created and saved {len(documents)} embeddings to {CHROMA_PATH}")
    return collection

def search_news(collection, openai_client, query, k=10):
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[query])
    query_embedding = resp.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return results

def rank_by_interest(results):
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ranked = []
    for doc, meta in zip(documents, metadatas):
        score = sum(1 for kw in LAW_KEYWORDS if kw.lower() in (doc or "").lower())
        date_str = (meta or {}).get("date", "")
        if date_str and any(y in date_str for y in ("2024", "2025")):
            score += 2
        ranked.append({"score": score, "doc": doc, "meta": meta or {}})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

# ========== MODEL SELECTION (Claude + OpenAI) ==========

def get_selected_claude_model() -> str:
    return st.session_state.get("CLAUDE_MODEL_SELECTED", CLAUDE_MODEL_EXPENSIVE)

def get_selected_openai_model() -> str:
    return st.session_state.get("OPENAI_MODEL_SELECTED", OPENAI_MODEL_EXPENSIVE)

def model_selector_sidebar():
    with st.sidebar.expander("⚙️ Model Settings", expanded=True):
        # Claude selector
        claude_choice = st.radio(
            "Claude chat model",
            [
                f"Expensive • {CLAUDE_MODEL_EXPENSIVE}",
                f"Cheaper • {CLAUDE_MODEL_CHEAP}"
            ],
            index=0 if st.session_state.get("CLAUDE_MODEL_SELECTED", CLAUDE_MODEL_EXPENSIVE) == CLAUDE_MODEL_EXPENSIVE else 1,
            key="model_choice_claude"
        )
        st.session_state["CLAUDE_MODEL_SELECTED"] = (
            CLAUDE_MODEL_EXPENSIVE if "Expensive" in claude_choice else CLAUDE_MODEL_CHEAP
        )
        st.caption(f"Claude using: `{st.session_state['CLAUDE_MODEL_SELECTED']}`")

        # OpenAI selector
        openai_choice = st.radio(
            "OpenAI chat model",
            [
                f"Expensive • {OPENAI_MODEL_EXPENSIVE}",
                f"Cheaper • {OPENAI_MODEL_CHEAP}"
            ],
            index=0 if st.session_state.get("OPENAI_MODEL_SELECTED", OPENAI_MODEL_EXPENSIVE) == OPENAI_MODEL_EXPENSIVE else 1,
            key="model_choice_openai"
        )
        st.session_state["OPENAI_MODEL_SELECTED"] = (
            OPENAI_MODEL_EXPENSIVE if "Expensive" in openai_choice else OPENAI_MODEL_CHEAP
        )
        st.caption(f"OpenAI using: `{st.session_state['OPENAI_MODEL_SELECTED']}`")

# ========== PAGES ==========

def page():
    st.title("🤖 News Bot for Law Firms ")
    st.caption("Using Claude for chat + OpenAI embeddings")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        claude_client = get_anthropic_client()
        openai_client = get_openai_client()

        if "collection" not in st.session_state:
            chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
            try:
                existing_count = collection.count()
            except Exception:
                existing_count = 0

            if existing_count > 0:
                st.session_state.collection = collection
                st.session_state.openai_client = openai_client
                st.success(f"✅ Loaded existing database ({existing_count} documents)")
            else:
                articles = load_csv_to_dict()
                articles = enrich_articles(articles)
                st.session_state.collection = create_vector_db(articles, openai_client)
                st.session_state.openai_client = openai_client
                st.success("✅ Database created and ready!")
        else:
            if "openai_client" not in st.session_state:
                st.session_state.openai_client = openai_client
            collection = st.session_state.collection
            openai_client = st.session_state.openai_client

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat
    if prompt := st.chat_input("Ask about the news..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if "most interesting" in prompt.lower():
                results = search_news(
                    collection,
                    openai_client,
                    "litigation regulation merger acquisition lawsuit compliance",
                    k=20
                )
                ranked = rank_by_interest(results)
                response = "**Top 10 Most Interesting News:**\n\n"
                for i, item in enumerate(ranked[:10], 1):
                    title = (item["doc"] or "")[:150] + "..."
                    url = item["meta"].get("url", "")
                    date = item["meta"].get("date", "")
                    score = item["score"]
                    response += f"{i}. [{title}]({url})\n"
                    response += f"   *Interest Score: {score} | {date}*\n\n"

            elif prompt.lower().startswith("find news about"):
                topic = prompt.replace("find news about", "").strip()
                results = search_news(collection, openai_client, topic, k=10)
                response = f"**News about '{topic}':**\n\n"
                for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                    title = (doc or "")[:150] + "..."
                    url = meta.get("url", "")
                    date = meta.get("date", "")
                    response += f"{i}. [{title}]({url})\n"
                    response += f"   *{date}*\n\n"

            else:
                # Use the selected Claude model for chat answers
                selected_model = get_selected_claude_model()
                results = search_news(collection, openai_client, prompt, k=5)
                context = "\n\n".join([f"Article: {(doc or '')[:500]}" for doc in results["documents"][0]])
                msg = claude_client.messages.create(
                    model=selected_model,
                    max_tokens=1024,
                    system="You are a news assistant for a law firm. Be concise and highlight legal implications.",
                    messages=[{"role": "user", "content": f"Question: {prompt}\n\nContext:\n{context}"}]
                )
                response = msg.content[0].text

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def test_page_openai():
    st.title("🧪 Testing Dashboard (OpenAI)")
    if "collection" not in st.session_state:
        st.warning("Collection not initialized yet. Open the Chat page first.")
        return

    collection = st.session_state.collection
    client = get_openai_client()
    selected_openai_model = get_selected_openai_model()

    if st.button("▶️ Run All Tests", type="primary"):
        with st.spinner("Running all tests..."):
            # Test 1
            st.header("Test 1: Most Interesting News")
            results = search_news(collection, client, "litigation regulation merger acquisition lawsuit compliance", k=20)
            ranked = rank_by_interest(results)

            st.subheader("Top 5 Results:")
            for i, item in enumerate(ranked[:5], 1):
                st.write(f"**{i}. Score: {item['score']}**")
                st.write(f"Date: {item['meta'].get('date', '')}")
                st.write(f"Preview: {(item['doc'] or '')[:100]}...")
                st.write(f"URL: {item['meta'].get('url', '')}")
                st.divider()

            top_5_scores = [item["score"] for item in ranked[:5]]
            avg_score = sum(top_5_scores) / max(1, len(top_5_scores))
            st.metric("Average Score (top 5)", f"{avg_score:.2f}", delta="Target: 3.0+")

            st.divider()

            # Test 2
            st.header("Test 2: Specific Topic Search")
            topics = ["antitrust", "merger", "regulation", "patent"]
            precisions = []
            for topic in topics:
                results = search_news(collection, client, topic, k=10)
                documents = results["documents"][0]
                relevant_count = sum(1 for doc in documents if doc and (topic.lower() in doc.lower()))
                precision = relevant_count / 10
                precisions.append(precision)
                st.write(f"**{topic}:** Precision = {precision*100:.0f}% ({relevant_count}/10 relevant)")

            avg_precision = (sum(precisions) / len(precisions)) if precisions else 0.0
            st.metric("Average Precision", f"{avg_precision*100:.1f}%", delta="Target: 80%+")

            st.divider()

            # Test 3 (OpenAI chat) — uses selected OpenAI model
            st.header("Test 3: General RAG Questions (OpenAI chat)")
            questions = [
                "What are the main legal issues in the news?",
                "Which companies are facing lawsuits?",
            ]
            for q in questions:
                st.subheader(f"Q: {q}")
                results = search_news(collection, client, q, k=5)
                context = "\n\n".join([f"Article: {(doc or '')[:300]}" for doc in results["documents"][0]])
                completion = client.chat.completions.create(
                    model=selected_openai_model,
                    messages=[
                        {"role": "system", "content": "You are a news assistant for a law firm. Be concise and highlight legal implications."},
                        {"role": "user", "content": f"Question: {q}\n\nContext:\n{context}"}]
                )
                answer = completion.choices[0].message.content
                st.write("**Answer:**")
                st.write(answer)
                st.divider()

            st.success("✅ All tests complete!")
        return

    st.info("Click 'Run All Tests' above to execute all test suites at once")
    st.divider()

def test_page_claude():
    """Claude-oriented test page mirroring OpenAI tests."""
    st.title("🧪 Testing Dashboard (Claude)")
    if "collection" not in st.session_state:
        st.warning("Collection not initialized yet. Open the Chat page first.")
        return

    collection = st.session_state.collection
    openai_client = get_openai_client()
    claude_client = get_anthropic_client()
    selected_claude_model = get_selected_claude_model()

    if st.button("▶️ Run All Tests (Claude)", type="primary"):
        with st.spinner("Running all tests..."):
            # Test 1
            st.header("Test 1: Most Interesting News")
            results = search_news(collection, openai_client, "litigation regulation merger acquisition lawsuit compliance", k=20)
            ranked = rank_by_interest(results)

            st.subheader("Top 5 Results:")
            for i, item in enumerate(ranked[:5], 1):
                st.write(f"**{i}. Score: {item['score']}**")
                st.write(f"Date: {item['meta'].get('date', '')}")
                st.write(f"Preview: {(item['doc'] or '')[:100]}...")
                st.write(f"URL: {item['meta'].get('url', '')}")
                st.divider()

            top_5_scores = [item["score"] for item in ranked[:5]]
            avg_score = sum(top_5_scores) / max(1, len(top_5_scores))
            st.metric("Average Score (top 5)", f"{avg_score:.2f}", delta="Target: 3.0+")

            st.divider()

            # Test 2
            st.header("Test 2: Specific Topic Search")
            topics = ["antitrust", "merger", "regulation", "patent"]
            precisions = []
            for topic in topics:
                results = search_news(collection, openai_client, topic, k=10)
                documents = results["documents"][0]
                relevant_count = sum(1 for doc in documents if doc and (topic.lower() in doc.lower()))
                precision = relevant_count / 10
                precisions.append(precision)
                st.write(f"**{topic}:** Precision = {precision*100:.0f}% ({relevant_count}/10 relevant)")

            avg_precision = (sum(precisions) / len(precisions)) if precisions else 0.0
            st.metric("Average Precision", f"{avg_precision*100:.1f}%", delta="Target: 80%+")

            st.divider()

            # Test 3 (Claude chat) — uses selected Claude model
            st.header("Test 3: General RAG Questions (Claude chat)")
            questions = [
                "What are the main legal issues in the news?",
                "Which companies are facing lawsuits?",
                "Summarize the merger activity"
            ]
            for q in questions:
                st.subheader(f"Q: {q}")
                results = search_news(collection, openai_client, q, k=5)
                context = "\n\n".join([f"Article: {(doc or '')[:300]}" for doc in results["documents"][0]])

                msg = claude_client.messages.create(
                    model=selected_claude_model,
                    max_tokens=512,
                    system="You are a news assistant for a law firm. Be concise and highlight legal implications.",
                    messages=[{"role": "user", "content": f"Question: {q}\n\nContext:\n{context}"}]
                )
                answer = msg.content[0].text

                st.write("**Answer:**")
                st.write(answer)
                st.write("**Context Used:**")
                for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                    st.write(f"{i}. {meta.get('date','')} - {(doc or '')[:80]}...")
                st.divider()

            st.success("✅ All tests complete!")
        return

    st.info("Click 'Run All Tests (Claude)' to execute all test suites at once")
    st.divider()

# ========== OPTIONAL TEST STUB ==========

def run_all_tests():
    """Safe stub to avoid NameError when called in __main__."""
    return None

# ========== ROUTER ==========

def run():
    # Model selectors (cheap vs expensive for both providers)
    model_selector_sidebar()

    # Navigation
    nav1 = st.sidebar.radio(
        "Navigation",
        ["💬 Chat", "🧪 Tests (OpenAI)", "🧪 Tests (Claude)"],
        key="nav_radio"
    )

    if nav1 == "💬 Chat":
        page()
    elif nav1 == "🧪 Tests (OpenAI)":
        test_page_openai()
    else:
        test_page_claude()

if __name__ == "__main__":
    run()
