import os
import csv
import streamlit as st
from openai import OpenAI
import chromadb
import requests
from bs4 import BeautifulSoup
import re

# ========== CONFIG ==========
CSV_PATH = "./files/Examples.csv"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "news"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

LAW_KEYWORDS = [
    "lawsuit", "litigation", "regulation", "antitrust", "merger", 
    "acquisition", "compliance", "patent", "settlement"
]

# ========== HELPER FUNCTIONS ==========

def get_openai_client():
    """Initialize OpenAI client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Set OPENAI_API_KEY environment variable")
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

def create_vector_db(articles, client, collection):
    """Build embeddings and add to the provided collection"""
    st.info("Creating embeddings for the first time...")
    
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
    
    batch_size = 100
    progress_bar = st.progress(0)
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        response = client.embeddings.create(
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
        
        progress_bar.progress(min((i + batch_size) / len(documents), 1.0))
    
    return collection

def search_news(collection, client, query, k=10):
    """Search for relevant news using vector similarity"""
    response = client.embeddings.create(
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
        score = sum(1 for keyword in LAW_KEYWORDS if keyword.lower() in doc.lower())
        
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

# ========== PAGES ==========

def chat_page():
    st.title("📰 News Bot for Law Firms (OpenAI)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
            collection = st.session_state.collection
            client = get_openai_client()
            
            if "most interesting" in prompt.lower():
                results = search_news(
                    collection, 
                    client, 
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
                
                results = search_news(collection, client, topic, k=10)
                
                response = f"**News about '{topic}':**\n\n"
                for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                    title = doc[:150] + "..."
                    url = meta.get("url", "")
                    date = meta.get("date", "")
                    
                    response += f"{i}. [{title}]({url})\n"
                    response += f"   *{date}*\n\n"
            
            else:
                results = search_news(collection, client, prompt, k=5)
                
                context = "\n\n".join([
                    f"Article: {doc[:500]}" 
                    for doc in results["documents"][0]
                ])
                
                messages = [
                    {"role": "system", "content": "You are a news assistant for a law firm. Be concise and highlight legal implications."},
                    {"role": "user", "content": f"Question: {prompt}\n\nContext:\n{context}"}
                ]
                
                completion = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages
                )
                response = completion.choices[0].message.content
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def test_page():
    st.title("🧪 Testing Dashboard (OpenAI)")
    
    collection = st.session_state.collection
    client = get_openai_client()
    
    st.header("Test 1: Most Interesting News")
    if st.button("Run Test 1"):
        results = search_news(
            collection, 
            client, 
            "litigation regulation merger acquisition lawsuit compliance",
            k=20
        )
        ranked = rank_by_interest(results)
        
        st.subheader("Top 5 Results:")
        for i, item in enumerate(ranked[:5], 1):
            st.write(f"**{i}. Score: {item['score']}**")
            st.write(f"Date: {item['meta'].get('date', '')}")
            st.write(f"Preview: {item['doc'][:100]}...")
            st.write(f"URL: {item['meta'].get('url', '')}")
            st.divider()
        
        top_5_scores = [item["score"] for item in ranked[:5]]
        avg_score = sum(top_5_scores) / 5
        st.metric("Average Score (top 5)", f"{avg_score:.2f}", delta="Target: 3.0+")
    
    st.header("Test 2: Specific Topic Search")
    topics = ["antitrust", "merger", "regulation", "patent"]
    
    if st.button("Run Test 2"):
        for topic in topics:
            st.subheader(f"Topic: {topic}")
            results = search_news(collection, client, topic, k=10)
            
            documents = results["documents"][0]
            relevant_count = sum(1 for doc in documents if topic.lower() in doc.lower())
            precision = relevant_count / 10
            
            st.metric(f"Precision", f"{precision*100:.0f}%", delta=f"{relevant_count}/10 relevant")
            
            with st.expander("See results"):
                for i, doc in enumerate(documents[:3], 1):
                    st.write(f"{i}. {doc[:100]}...")
    
    st.header("Test 3: General RAG Questions")
    questions = [
        "What are the main legal issues in the news?",
        "Which companies are facing lawsuits?",
        "Summarize the merger activity"
    ]
    
    selected_q = st.selectbox("Select question", questions)
    if st.button("Run Test 3"):
        results = search_news(collection, client, selected_q, k=5)
        
        context = "\n\n".join([
            f"Article: {doc[:300]}" 
            for doc in results["documents"][0]
        ])
        
        messages = [
            {"role": "system", "content": "You are a news assistant for a law firm. Be concise and highlight legal implications."},
            {"role": "user", "content": f"Question: {selected_q}\n\nContext:\n{context}"}
        ]
        
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )
        answer = completion.choices[0].message.content
        
        st.write("**Answer:**")
        st.write(answer)
        
        st.write("**Context Used:**")
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
            st.write(f"{i}. {meta.get('date')} - {doc[:80]}...")

# ========== MAIN APP ==========

st.set_page_config(page_title="News Bot (OpenAI)", page_icon="📰", layout="wide")

# Initialize database
try:
    client = get_openai_client()
    
    if "collection" not in st.session_state:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        
        try:
            existing_count = collection.count()
        except:
            existing_count = 0
        
        if existing_count > 0:
            st.session_state.collection = collection
            st.success(f"✅ Loaded existing database ({existing_count} documents)")
        else:
            articles = load_csv_to_dict()
            articles = enrich_articles(articles)
            st.session_state.collection = create_vector_db(articles, client, collection)
            st.success("✅ Database created and ready!")
    
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Navigation
page = st.sidebar.radio("Navigation", ["💬 Chat", "🧪 Tests"])

if page == "💬 Chat":
    chat_page()
else:
    test_page()