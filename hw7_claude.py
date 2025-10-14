import datetime
import os
import csv
import streamlit as st
from anthropic import Anthropic
import chromadb
import requests
from bs4 import BeautifulSoup
import re

# ========== CONFIG ==========
CSV_PATH = "./files/Examples.csv"
CHROMA_PATH = "./chroma_db_claude"
COLLECTION_NAME = "news_claude"
CHAT_MODEL = "claude-3-5-sonnet-20241022"

# Note: Claude doesn't provide embeddings, so we'll use a simple embedding service
# For this demo, we'll use sentence-transformers locally
from sentence_transformers import SentenceTransformer

# Keywords for ranking "interesting" news (law firm context)
LAW_KEYWORDS = [
    "lawsuit", "litigation", "regulation", "antitrust", "merger", 
    "acquisition", "compliance", "patent", "settlement"
]

# ========== HELPER FUNCTIONS ==========

def get_anthropic_client():
    """Initialize Anthropic client"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Set ANTHROPIC_API_KEY environment variable")
        st.stop()
    return Anthropic(api_key=api_key)

def get_embedding_model():
    """Load sentence transformer for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

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

def create_vector_db(articles, embed_model):
    """Create ChromaDB collection with embeddings"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    
    if collection.count() > 0:
        return collection
    
    st.info("Creating embeddings...")
    
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
    
    # Create embeddings in batches
    batch_size = 100
    progress_bar = st.progress(0)
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Use sentence-transformers for embeddings
        embeddings = embed_model.encode(batch_docs).tolist()
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
            ids=batch_ids
        )
        
        progress_bar.progress((i + batch_size) / len(documents))
    
    return collection

def search_news(collection, embed_model, query, k=10):
    """Search for relevant news using vector similarity"""
    query_embedding = embed_model.encode([query])[0].tolist()
    
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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load data and create vector DB
    try:
        client = get_anthropic_client()
        
        if "collection" not in st.session_state:
            embed_model = get_embedding_model()
            articles = load_csv_to_dict()
            articles = enrich_articles(articles)
            st.session_state.collection = create_vector_db(articles, embed_model)
            st.session_state.embed_model = embed_model
            st.success("✅ Ready!")
        
        collection = st.session_state.collection
        embed_model = st.session_state.embed_model
        
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
                    embed_model, 
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
                
                results = search_news(collection, embed_model, topic, k=10)
                
                response = f"**News about '{topic}':**\n\n"
                for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                    title = doc[:150] + "..."
                    url = meta.get("url", "")
                    date = meta.get("date", "")
                    
                    response += f"{i}. [{title}]({url})\n"
                    response += f"   *{date}*\n\n"
            
            else:
                results = search_news(collection, embed_model, prompt, k=5)
                
                context = "\n\n".join([
                    f"Article: {doc[:500]}" 
                    for doc in results["documents"][0]
                ])
                
                # Use Claude API
                message = client.messages.create(
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


# ========== TEST QUERIES ==========
TEST_QUERIES = {
    "most_interesting": [
        "find the most interesting news",
    ],
    "specific_topic": [
        "find news about antitrust",
        "find news about merger",
        "find news about regulation",
        "find news about patent",
    ],
    "general_rag": [
        "What are the main legal issues in the news?",
        "Which companies are facing lawsuits?",
        "Summarize the merger activity",
    ]
}

# ========== HELPER FUNCTIONS ==========

def get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")
    return Anthropic(api_key=api_key)

def get_embedding_model():
    """Load sentence transformer for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_collection():
    """Load existing ChromaDB collection"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    return collection

def search_news(collection, embed_model, query, k=10):
    query_embedding = embed_model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    return results

def rank_by_interest(results):
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

def run_most_interesting_test(collection, embed_model):
    """Test: find the most interesting news"""
    print("\n" + "="*80)
    print("TEST 1: MOST INTERESTING NEWS (CLAUDE)")
    print("="*80)
    
    results = search_news(
        collection, 
        embed_model, 
        "litigation regulation merger acquisition lawsuit compliance",
        k=20
    )
    ranked = rank_by_interest(results)
    
    print("\nTop 10 Results:")
    for i, item in enumerate(ranked[:10], 1):
        title = item["doc"][:100] + "..."
        score = item["score"]
        date = item["meta"].get("date", "")
        url = item["meta"].get("url", "")
        
        print(f"\n{i}. Score: {score} | Date: {date}")
        print(f"   {title}")
        print(f"   URL: {url}")
    
    # Calculate metrics
    top_5_scores = [item["score"] for item in ranked[:5]]
    avg_score = sum(top_5_scores) / len(top_5_scores)
    
    recent_count = sum(1 for item in ranked[:10] 
                       if "2024" in item["meta"].get("date", "") 
                       or "2025" in item["meta"].get("date", ""))
    
    print("\n" + "-"*80)
    print("METRICS:")
    print(f"  Average score (top 5): {avg_score:.2f}")
    print(f"  Recent articles (top 10): {recent_count}/10")
    print(f"  Min score needed: 3.0 (target)")
    print(f"  Recent target: 8/10")
    
    # Manual evaluation prompts
    print("\n" + "-"*80)
    print("MANUAL EVALUATION:")
    print("  [ ] Are top 5 results actually law-firm relevant?")
    print("  [ ] Do top 5 contain 2+ law keywords each?")
    print("  [ ] Would a law partner find these interesting?")
    print("  [ ] Any obvious legal stories ranked too low?")
    
    return ranked[:10]

def run_specific_topic_test(collection, embed_model, topic_query):
    """Test: find news about specific topic"""
    print("\n" + "="*80)
    print(f"TEST 2: SPECIFIC TOPIC - '{topic_query}' (CLAUDE)")
    print("="*80)
    
    topic = topic_query.replace("find news about", "").strip()
    results = search_news(collection, embed_model, topic, k=10)
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    print(f"\nTop 10 Results for '{topic}':")
    relevant_count = 0
    
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        title = doc[:100] + "..."
        date = meta.get("date", "")
        url = meta.get("url", "")
        
        # Check if topic appears in doc
        topic_in_doc = topic.lower() in doc.lower()
        if topic_in_doc:
            relevant_count += 1
        
        print(f"\n{i}. Date: {date} | Relevant: {'✓' if topic_in_doc else '✗'}")
        print(f"   {title}")
        print(f"   URL: {url}")
    
    precision = relevant_count / 10
    
    print("\n" + "-"*80)
    print("METRICS:")
    print(f"  Precision: {relevant_count}/10 ({precision*100:.0f}%)")
    print(f"  Target: ≥8/10 (80%)")
    
    print("\n" + "-"*80)
    print("MANUAL EVALUATION:")
    print("  [ ] Are results actually about the topic?")
    print("  [ ] Did it find semantically related content?")
    print("  [ ] Any obvious relevant articles missing?")
    
    return precision

def run_general_rag_test(collection, embed_model, client, question):
    """Test: general RAG question"""
    print("\n" + "="*80)
    print(f"TEST 3: GENERAL RAG - '{question}' (CLAUDE)")
    print("="*80)
    
    results = search_news(collection, embed_model, question, k=5)
    
    context = "\n\n".join([
        f"Article: {doc[:300]}" 
        for doc in results["documents"][0]
    ])
    
    # Use Claude API
    message = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        system="You are a news assistant for a law firm. Be concise and highlight legal implications.",
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}"
            }
        ]
    )
    answer = message.content[0].text
    
    print("\nGENERATED ANSWER:")
    print(answer)
    
    print("\n" + "-"*80)
    print("CONTEXT ARTICLES USED:")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        print(f"{i}. {meta.get('date')} - {doc[:80]}...")
    
    print("\n" + "-"*80)
    print("MANUAL EVALUATION:")
    print("  [ ] Does answer cite specific articles?")
    print("  [ ] Is information factually accurate?")
    print("  [ ] No hallucinated facts?")
    print("  [ ] Quality rating (1-5): ___")
    
    return answer

def run_all_tests():
    """Run all test suites"""
    print("\n" + "#"*80)
    print("# NEWS BOT TESTING SCRIPT - CLAUDE VERSION")
    print("# " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#"*80)
    
    try:
        client = get_anthropic_client()
        embed_model = get_embedding_model()
        collection = load_collection()
        
        print(f"\nLoaded collection: {collection.count()} documents")
        print(f"Using model: {CHAT_MODEL}")
        print(f"Using embeddings: sentence-transformers (all-MiniLM-L6-v2)")
        
        # Test 1: Most Interesting
        results_interesting = run_most_interesting_test(collection, embed_model)
        
        # Test 2: Specific Topics
        precisions = []
        for query in TEST_QUERIES["specific_topic"]:
            precision = run_specific_topic_test(collection, embed_model, query)
            precisions.append(precision)
        
        avg_precision = sum(precisions) / len(precisions)
        
        # Test 3: General RAG
        for query in TEST_QUERIES["general_rag"]:
            run_general_rag_test(collection, embed_model, client, query)
        
        # Final Summary
        print("\n" + "#"*80)
        print("# SUMMARY")
        print("#"*80)
        print(f"\nAverage Precision (specific topics): {avg_precision*100:.1f}%")
        print(f"Target: ≥80%")
        print(f"Status: {'✓ PASS' if avg_precision >= 0.8 else '✗ FAIL'}")
        
        print("\n" + "#"*80)
        print("# NEXT STEPS")
        print("#"*80)
        print("1. Review flagged items above")
        print("2. Complete manual evaluations")
        print("3. Compare with OpenAI version results")
        print("4. Compare answer quality between models")
        print("5. Document findings")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure:")
        print("  1. ANTHROPIC_API_KEY is set")
        print("  2. ChromaDB collection exists (run Claude app first)")
        print("  3. sentence-transformers is installed")

def run():
    page()



if __name__ == "__main__":
    run()
    run_all_tests()
