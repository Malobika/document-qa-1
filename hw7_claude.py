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

# Keywords for ranking "interesting" news (law firm context)
LAW_KEYWORDS = [
    "lawsuit", "litigation", "regulation", "antitrust", "merger", 
    "acquisition", "compliance", "patent", "settlement"
]

# ========== HELPER FUNCTIONS ==========

def get_anthropic_client():
    """Initialize Anthropic client"""
    api_key = os.getenv("CLAUDE_API_KEY")
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
    """Fetch and extract text from URL using BeautifulSoup"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()[:10000]
    except:
        pass
    return None

def load_csv_to_dict():
    """Load CSV as list of dicts"""
    articles = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles.append(row)
    return articles

def enrich_articles(articles):
    """Fetch URL content for each article"""
    st.info(f"Loading {len(articles)} articles...")
    progress_bar = st.progress(0)
    
    for i, article in enumerate(articles):
        url = article.get("URL", "")
        if url:
            fetched_text = fetch_url_text(url)
            if fetched_text:
                article["Document"] = fetched_text
        progress_bar.progress((i + 1) / len(articles))
    
    return articles

def create_vector_db(articles, openai_client):
    """Create ChromaDB collection with OpenAI embeddings"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    
    if collection.count() > 0:
        return collection
    
    st.info("Creating embeddings with OpenAI...")
    
    documents = []
    metadatas = []
    ids = []
    
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
    
    # Create embeddings in batches using OpenAI
    batch_size = 100
    progress_bar = st.progress(0)
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Use OpenAI embeddings
        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=batch_docs
        )
        embeddings = [item.embedding for item in response.data]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
            ids=batch_ids
        )
        
        progress_bar.progress((i + batch_size) / len(documents))
    
    return collection

def search_news(collection, openai_client, query, k=10):
    """Search for relevant news using OpenAI embeddings"""
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query]
    )
    query_embedding = response.data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    return results

def rank_by_interest(results):
    """Rank results by law firm interest level"""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    ranked = []
    for doc, meta in zip(documents, metadatas):
        # Score based on law keywords
        score = sum(1 for keyword in LAW_KEYWORDS if keyword.lower() in doc.lower())
        
        # Add recency boost
        date_str = meta.get("date", "")
        if date_str:
            if "2024" in date_str or "2025" in date_str:
                score += 2
        
        ranked.append({
            "score": score,
            "doc": doc,
            "meta": meta
        })
    
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

# ========== STREAMLIT UI ==========
def page():
    st.set_page_config(page_title="News Bot (Claude)", page_icon="🤖")
    st.title("🤖 News Bot for Law Firms (Claude)")
    st.caption("Using Claude for chat + OpenAI embeddings")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load data and create vector DB
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

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
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
                    title = item["doc"][:150] + "..."
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
                    title = doc[:150] + "..."
                    url = meta.get("url", "")
                    date = meta.get("date", "")
                    
                    response += f"{i}. [{title}]({url})\n"
                    response += f"   *{date}*\n\n"
            
            else:
                results = search_news(collection, openai_client, prompt, k=5)
                
                context = "\n\n".join([
                    f"Article: {doc[:500]}" 
                    for doc in results["documents"][0]
                ])
                
                # Use Claude API
                message = claude_client.messages.create(
                    model=CHAT_MODEL,
                    max_tokens=1024,
                    system="You are a news assistant for a law firm. Be concise and highlight legal implications.",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Question: {prompt}\n\nContext:\n{context}"
                        }
                    ]
                )
                response = message.content[0].text
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


def run():
    page()



if __name__ == "__main__":
    run()
    run_all_tests()