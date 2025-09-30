# HWs/HW5.py
# ---------------------------------------------
# HW5: RAG chatbot with FUNCTION-BASED retrieval
# - Instead of embedding context in prompt, LLM calls a function to retrieve info
# - Short-term memory: keeps last 5 Q&A pairs
# - Uses the same vector DB from HW4 (Chroma)
# - Supports multiple LLM backends via sidebar

import os
import re
import json
from typing import List, Dict, Any
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# --- Chroma setup
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.errors import IDAlreadyExistsError

load_dotenv()

# -------- Configuration --------
HTML_FOLDER = "./su_orgs"  # Your HTML files location
CHROMA_PATH = "./Chroma_HW4_new"  # Reuse HW4's vector DB
COLLECTION_NAME = "HW4_HTML_Collection"
EMBED_MODEL = "text-embedding-3-small"

# -------- Utility Functions --------
def _safe_html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

def _list_html_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".html")]

def _chunk_into_two(text: str) -> tuple:
    """Split text into exactly two chunks at paragraph boundaries."""
    if not text:
        return "", ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:
        mid = max(1, len(text)//2)
        return text[:mid].strip(), text[mid:].strip()
    
    total_len = sum(len(p) for p in paras)
    target = total_len // 2
    acc, chunk1 = 0, []
    for p in paras:
        if acc < target:
            chunk1.append(p)
            acc += len(p)
        else:
            break
    
    chunk2_start_idx = len(chunk1)
    chunk1_text = "\n\n".join(chunk1).strip()
    chunk2_text = "\n\n".join(paras[chunk2_start_idx:]).strip()
    return chunk1_text, chunk2_text

def _ensure_vector_db(openai_client: OpenAI):
    """Load or create the persistent Chroma collection."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    
    try:
        existing_count = collection.count()
    except Exception:
        existing_count = 0
    
    if existing_count > 0:
        return collection
    
    # Build vector DB if empty
    html_files = _list_html_files(HTML_FOLDER)
    if not html_files:
        st.warning(f"No HTML files found in {HTML_FOLDER}.")
        return collection
    
    with st.status("Building vector DB from HTML files…", expanded=False):
        for path in html_files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
                text = _safe_html_to_text(html)
                c1, c2 = _chunk_into_two(text)
                
                for i, chunk in enumerate([c1, c2], start=1):
                    if not chunk:
                        continue
                    emb = openai_client.embeddings.create(
                        model=EMBED_MODEL,
                        input=chunk
                    ).data[0].embedding
                    doc_id = f"{os.path.basename(path)}::part{i}"
                    meta = {"filename": os.path.basename(path), "part": i}
                    try:
                        collection.add(documents=[chunk], embeddings=[emb], ids=[doc_id], metadatas=[meta])
                    except IDAlreadyExistsError:
                        pass
            except Exception as e:
                st.write(f"⚠ Skipped {os.path.basename(path)}: {e}")
        st.success("Vector DB created ✅")
    
    return collection

# -------- FUNCTION-BASED RETRIEVAL (Key HW5 Feature) --------
def get_relevant_club_info(query: str, collection, openai_client: OpenAI, k: int = 4) -> str:
    """
    Function that retrieves relevant information from vector DB.
    This is what the LLM will CALL instead of us pre-embedding context in the prompt.
    
    Args:
        query: Search query from the LLM
        collection: Chroma collection
        openai_client: OpenAI client for embeddings
        k: Number of chunks to retrieve
    
    Returns:
        Formatted string with retrieved context
    """
    # Get query embedding
    q_emb = openai_client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    
    # Query vector DB
    result = collection.query(query_embeddings=[q_emb], n_results=k)
    
    if not result or not result.get("documents"):
        return "No relevant information found."
    
    # Format results
    context_parts = []
    for i in range(len(result["documents"][0])):
        text = result["documents"][0][i]
        metadata = result["metadatas"][0][i]
        filename = metadata.get("filename", "unknown")
        part = metadata.get("part", "?")
        context_parts.append(f"[Source: {filename} part {part}]\n{text}")
    
    return "\n\n---\n\n".join(context_parts)

# Function schema for OpenAI function calling
FUNCTION_SCHEMA = {
    "name": "get_relevant_club_info",
    "description": "Searches the vector database for information about student organizations, clubs, and activities from the HTML documents. Use this when you need specific information to answer the user's question.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant information (e.g., 'engineering clubs', 'club meeting times', 'how to join organizations')"
            }
        },
        "required": ["query"]
    }
}

# -------- LLM INTERACTION WITH FUNCTION CALLING --------
def chat_with_function_calling(messages: List[Dict], openai_client: OpenAI, collection, model: str = "gpt-4o-mini", k: int = 4) -> str:
    """
    Main chat function that uses function calling for retrieval.
    
    Flow:
    1. Call LLM with function definition (no context yet)
    2. If LLM wants to call function, execute it
    3. Call LLM again with function results
    4. Return final answer
    """
    
    # First call: LLM decides if it needs to search
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        functions=[FUNCTION_SCHEMA],
        function_call="auto",
        temperature=0.3
    )
    
    message = response.choices[0].message
    
    # Check if LLM wants to call our function
    if message.function_call:
        function_name = message.function_call.name
        arguments = json.loads(message.function_call.arguments)
        
        # Execute the function
        if function_name == "get_relevant_club_info":
            search_query = arguments.get("query", "")
            st.info(f"🔍 Searching vector DB for: `{search_query}`")
            
            # Call our retrieval function
            context = get_relevant_club_info(search_query, collection, openai_client, k)
            
            # Add assistant's function call to conversation
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": json.dumps(arguments)
                }
            })
            
            # Add function result to conversation
            messages.append({
                "role": "function",
                "name": function_name,
                "content": context
            })
            
            # Second call: LLM generates answer using function results
            final_response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3
            )
            
            return final_response.choices[0].message.content
    
    # If no function call, just return the direct response
    return message.content if message.content else "I need more information to answer that question."

# -------- STREAMLIT PAGE --------
def page():
    st.title("🤖 HW5 – Function-Based RAG Chatbot")
    st.caption("Short-term memory chatbot using function calling for retrieval")
    
    # --- API key check ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No OpenAI API key found. Add OPENAI_API_KEY to your .env file.")
        st.stop()
    
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=api_key)
    
    client = st.session_state.openai_client
    
    # --- Load vector DB ---
    if "collection_hw5" not in st.session_state:
        st.session_state.collection_hw5 = _ensure_vector_db(client)
    
    collection = st.session_state.collection_hw5
    if collection is None:
        st.stop()
    
    # --- Sidebar settings ---
    st.sidebar.header("⚙️ Settings")
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0
    )
    top_k = st.sidebar.slider("Top-k chunks", min_value=2, max_value=8, value=4)
    
    st.sidebar.divider()
    st.sidebar.caption("💡 **How it works:**")
    st.sidebar.caption("1. You ask a question")
    st.sidebar.caption("2. LLM decides if it needs info")
    st.sidebar.caption("3. LLM calls function to search")
    st.sidebar.caption("4. LLM answers using results")
    
    # --- Initialize conversation memory (last 5 Q&A pairs) ---
    if "messages_hw5" not in st.session_state:
        st.session_state.messages_hw5 = []
    
    # System prompt (explain function calling capability)
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions about student organizations and clubs. "
            "When you need specific information from the documents, use the get_relevant_club_info function. "
            "Always cite sources in your answers using the format [Source: filename.html part X]. "
            "Be concise and helpful."
        )
    }
    
    # --- Display chat history ---
    for msg in st.session_state.messages_hw5:
        if msg["role"] in ["user", "assistant"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # --- User input ---
    user_input = st.chat_input("Ask about clubs, organizations, or activities...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages_hw5.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Maintain short-term memory (last 5 Q&A pairs = 10 messages max)
        # Keep only user/assistant messages for memory calculation
        conversation_messages = [m for m in st.session_state.messages_hw5 if m["role"] in ["user", "assistant"]]
        if len(conversation_messages) > 10:  # 5 Q&A pairs
            # Remove oldest Q&A pair
            st.session_state.messages_hw5 = st.session_state.messages_hw5[-10:]
        
        # Prepare messages for LLM (system + recent history + current question)
        llm_messages = [system_prompt] + st.session_state.messages_hw5
        
        # Get response with function calling
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_function_calling(
                    llm_messages,
                    client,
                    collection,
                    model=model,
                    k=top_k
                )
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages_hw5.append({"role": "assistant", "content": response})
    
    # --- Clear chat button ---
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages_hw5 = []
        st.rerun()
    
    # --- Info section ---
    with st.expander("ℹ️ About HW5"):
        st.markdown("""
        **Key Differences from HW4:**
        
        - **HW4**: Retrieved context and stuffed it into the prompt manually
        - **HW5**: LLM decides when to search and calls a function to retrieve info
        
        **Benefits:**
        - More efficient (only retrieves when needed)
        - LLM can formulate better search queries
        - Cleaner separation of concerns
        - More flexible conversation flow
        
        **Short-term Memory:**
        - Keeps last 5 question-answer pairs
        - Automatically trims older messages
        - Maintains conversation context
        """)

# Public entry point
def run():
    page()

if __name__ == "__main__":
    run()