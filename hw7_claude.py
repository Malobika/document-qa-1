# === FIXES & TEST PAGES (paste into your main .py) ===
import os
import csv
import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
import chromadb
import requests
from bs4 import BeautifulSoup
import re

# ========== CONFIG ==========
CSV_PATH = "./files/Examples.csv"
CHROMA_PATH = "./chroma_db_claude"
COLLECTION_NAME = "news_claude"
EMBED_MODEL = "text-embedding-3-small"  # Using OpenAI embeddings
CHAT_MODEL = "claude-3-5-sonnet-20241022"

LAW_KEYWORDS = [
    "lawsuit", "litigation", "regulation", "antitrust", "merger",
    "acquisition", "compliance", "patent", "settlement"
]

# ========== HELPERS ==========

def get_anthropic_client():
    """Initialize Anthropic client"""
    api_key = os.getenv("CLAUDE_API_KEY")  # <-- fix: canonical env var + message
    if not api_key:
        st.error("Set ANTHROPIC_API_KEY environment variable")
        st.stop()
    return Anthropic(api_key=api_key)

def get_openai_client():
    """Initialize OpenAI client for embeddings only"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Set OPENAI_API_KEY environment variable (needed for embeddings)")
        st.stop()
    return OpenAI(api_key=api_key)

def fetch_url_text(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()[:10000]
    except Exception:
        pass
    return None

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

def create_vector_db(articles, openai_client, collection=None):
    """
    Create/Populate ChromaDB collection with OpenAI embeddings.
    If `collection` is provided, add into it; otherwise create a persistent one.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = collection or chroma_client.get_or_create_collection(COLLECTION_NAME)

    st.info("Creating embeddings with OpenAI (first-time population or add)…")

    documents, metadatas, ids = [], [], []
    for i, article in enumerate(articles):
        doc_text = article.get("Document", "")
        if len(doc_text) > 50:
            documents.append(doc_text)
            metadatas.append({
                "company": article.get("company_name") or article.get("company") or "",
                "date": article.get("Date", ""),
                "url": article.get("URL", "")
            })
            ids.append(f"doc_{i}")

    if not documents:
        st.warning("No documents with sufficient content to embed.")
        return collection

    batch_size = 100
    progress_bar = st.progress(0)
    total = len(documents)

    for i in range(0, total, batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        resp = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=batch_docs
        )
        embeddings = [item.embedding for item in resp.data]

        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
            ids=batch_ids
        )
        progress_bar.progress(min((i + batch_size) / total, 1.0))

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

# ========== MAIN CLAUDE PAGE ==========

def page():
    st.set_page_config(page_title="News Bot (Claude)", page_icon="🤖")
    st.title("🤖 News Bot for Law Firms (Claude)")
    st.caption("Using Claude for chat + OpenAI embeddings")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        claude_client = get_anthropic_client()
        openai_client = get_openai_client()

        if "collection" not in st.session_state:
            articles = load_csv_to_dict()
            articles = enrich_articles(articles)
            st.session_state.collection = create_vector_db(articles, openai_client)
            st.session_state.openai_client = openai_client
            st.success("✅ Ready!")

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
                results = search_news(collection, openai_client, prompt, k=5)
                context = "\n\n".join([f"Article: {(doc or '')[:500]}" for doc in results["documents"][0]])
                # Claude for the answer
                message = claude_client.messages.create(
                    model=CHAT_MODEL,
                    max_tokens=1024,
                    system="You are a news assistant for a law firm. Be concise and highlight legal implications.",
                    messages=[{"role": "user", "content": f"Question: {prompt}\n\nContext:\n{context}"}]
                )
                response = message.content[0].text

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ========= STREAMLIT TEST PAGES =========

def test_page_openai():
    # Your existing OpenAI test page (unchanged logic), ensure CHAT_MODEL is an OpenAI model if used here
    st.title("🧪 Testing Dashboard (OpenAI)")
    collection = st.session_state.collection
    client = get_openai_client()

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

            # Test 3 (OpenAI model only if you have a valid chat model; otherwise skip)
            st.header("Test 3: General RAG Questions (OpenAI chat)")
            questions = [
                "What are the main legal issues in the news?",
                "Which companies are facing lawsuits?",
            ]
            for q in questions:
                st.subheader(f"Q: {q}")
                results = search_news(collection, client, q, k=5)
                context = "\n\n".join([f"Article: {(doc or '')[:300]}" for doc in results["documents"][0]])
                # If you want to keep this, set an OpenAI chat model here:
                OPENAI_CHAT_MODEL = "gpt-4o-mini"
                completion = client.chat.completions.create(
                    model=OPENAI_CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a news assistant for a law firm. Be concise and highlight legal implications."},
                        {"role": "user", "content": f"Question: {q}\n\nContext:\n{context}"}
                    ]
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
    """New: Claude-oriented test page mirroring your OpenAI tests"""
    st.title("🧪 Testing Dashboard (Claude)")
    collection = st.session_state.collection
    openai_client = get_openai_client()
    claude_client = get_anthropic_client()

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

            # Test 3 (Claude chat)
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
                    model=CHAT_MODEL,
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

# ========== SIMPLE NAV ==========


def run_all_tests():
    """Safe stub to avoid NameError when called in __main__.
    For CLI tests run `pytest -q` instead."""
    return None

# ========== MAIN APP ==========
# ========== MAIN APP ==========
st.set_page_config(page_title="News Bot (OpenAI/Claude)", page_icon="📰", layout="wide")

try:
    openai_client = get_openai_client()
    # ✅ FIX: always store the client in session state
    st.session_state.openai_client = openai_client

    if "collection" not in st.session_state:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        try:
            existing_count = collection.count()
        except Exception:
            existing_count = 0

        if existing_count > 0:
            st.session_state.collection = collection
            st.success(f"✅ Loaded existing database ({existing_count} documents)")
        else:
            articles = load_csv_to_dict()
            articles = enrich_articles(articles)
            st.session_state.collection = create_vector_db(articles, openai_client, collection)
            st.success("✅ Database created and ready!")

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()


page_choice = st.sidebar.radio("Navigation", ["💬 Chat", "🧪 Tests (OpenAI)", "🧪 Tests (Claude)"])

if page_choice == "💬 Chat":
    page()
elif page_choice == "🧪 Tests (OpenAI)":
    test_page_openai()
else:
    test_page_claude()



def run():
    # Sidebar navigation
    nav = st.sidebar.radio(
        "Navigation",
        ["💬 Chat", "🧪 Tests (OpenAI)", "🧪 Tests (Claude)"]
    )

    if nav == "💬 Chat":
        # Use chat_page() if you defined it; otherwise fall back to page()
        try:
            page()
        except NameError:
            page()

    elif nav == "🧪 Tests (OpenAI)":
        # Your existing OpenAI test dashboard
        test_page_openai()  # or test_page_openai() if you split them

    else:
        # Claude-oriented test dashboard
        test_page_claude()

if __name__ == "__main__":
    run()
