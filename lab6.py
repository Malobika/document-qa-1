import os
import re
import time
import pandas as pd
import streamlit as st

from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Lab 6 — ReAct Research Assistant", page_icon="🔎", layout="wide")
st.title("🔎 Lab 6 — LangChain ReAct Research Assistant (arXiv)")

st.caption(
    "Search and compare research papers with a ReAct agent, custom tools, and a persistent Chroma vector DB."
)


# ---------------------------
# Session state init
# ---------------------------
def _init_state():
    defaults = {
        "lab6_messages": [],
        "lab6_df": None,
        "lab6_vectorstore": None,
        "lab6_agent": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------
# Sidebar: keys & info
# ---------------------------
with st.sidebar:
    st.header("Settings")
    st.markdown(
        "Add your API key via **.streamlit/secrets.toml** as:\n\n"
        "```toml\n[general]\nopenai_api_key = \"sk-...\"\n```\n"
        "On Streamlit Community Cloud, set it under *App settings → Secrets*."
    )

    if "openai_api_key" not in st.secrets.get("general", {}):
        st.warning("OpenAI API key not found in secrets. Set [general].openai_api_key to use the app.")

    st.divider()
    st.caption("CSV file expected columns: title, authors, abstract, year, category, venue, link")


# ---------------------------
# Vectorstore setup
# ---------------------------
@st.cache_resource
def initialize_vectorstore():
    """
    Initialize a persistent Chroma DB from a local CSV of arXiv papers.
    The CSV is expected to have columns:
    title, authors, abstract, year, category, venue, link
    """
    CSV_PATH = "papers.csv"
    PERSIST_DIR = "LAB6_vector_db"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at {CSV_PATH}. Please place your dataset in the app directory."
        )

    os.makedirs(PERSIST_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Build documents with metadata (important since tools read metadata)
    docs = []
    for _, row in df.iterrows():
        meta = {
            "title": row.get("title", ""),
            "authors": row.get("authors", ""),
            "year": row.get("year", ""),
            "category": row.get("category", ""),
            "venue": row.get("venue", ""),
            "link": row.get("link", ""),
        }
        text = (
            f"Title: {meta['title']}\n"
            f"Authors: {meta['authors']}\n"
            f"Abstract: {row.get('abstract','')}\n"
            f"Year: {meta['year']}\n"
            f"Category: {meta['category']}\n"
            f"Venue: {meta['venue']}\n"
            f"Link: {meta['link']}"
        )
        docs.append(Document(page_content=text, metadata=meta))

    embeddings = OpenAIEmbeddings(
        api_key=st.secrets["general"].get("openai_api_key", None),
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    return vectorstore, df


# Initialize vectorstore and data (persisted)
if st.session_state.lab6_vectorstore is None:
    try:
        with st.spinner("Building / loading vector database..."):
            vs, df = initialize_vectorstore()
            st.session_state.lab6_vectorstore = vs
            st.session_state.lab6_df = df
    except Exception as e:
        st.error(f"Vectorstore Error: {e}")
        st.stop()


# ---------------------------
# Tool definitions
# ---------------------------
def search_papers(query: str) -> str:
    """
    Uses the Chroma vectorstore to semantic-search for papers related to `query`.
    Returns a formatted string with top-k results.
    """
    results = st.session_state.lab6_vectorstore.similarity_search(query, k=5)
    if not results:
        return f"No papers found about '{query}'."

    lines = []
    for i, doc in enumerate(results, start=1):
        meta = doc.metadata or {}
        title = meta.get("title", "")
        authors = meta.get("authors", "")
        link = meta.get("link", "")
        lines.append(f"{i}. {title}\nAuthors: {authors}\nLink: {link}\n")
    return "\n".join(lines)


def compare_papers(query: str) -> str:
    """
    Compares two papers by approximate title matching in the loaded CSV.
    User input format examples:
      - 'Attention Is All You Need and BERT: Pre-training of Deep ...'
      - 'Paper A vs. Paper B'
    """
    parts = re.split(r"\s+and\s+|\s+vs\.?\s+", query, flags=re.IGNORECASE)
    if len(parts) < 2:
        return "Please specify two papers: e.g., 'paper1 and paper2' or 'paper1 vs. paper2'."

    df = st.session_state.lab6_df

    def find(title_fragment: str):
        mask = df["title"].fillna("").str.contains(title_fragment.strip(), case=False, na=False)
        if not mask.any():
            return None
        r = df[mask].iloc[0]
        snippet = (r.get("abstract", "") or "")[:500]
        return (
            f"**Title:** {r.get('title','')}\n"
            f"**Authors:** {r.get('authors','')}\n"
            f"**Year:** {r.get('year','')}\n"
            f"**Category:** {r.get('category','')}\n"
            f"**Venue:** {r.get('venue','')}\n"
            f"**Link:** {r.get('link','')}\n\n"
            f"**Abstract (truncated):** {snippet}..."
        )

    p1, p2 = find(parts[0]), find(parts[1])
    if not p1 or not p2:
        return "Could not find one or both papers. Try providing more distinct title text."

    return f"### Paper 1\n{p1}\n\n### Paper 2\n{p2}"


tools = [
    Tool(
        name="SearchPapers",
        func=search_papers,
        description="Find research papers on a topic. Input should be a topical query, e.g., 'graph transformers for chemistry'."
    ),
    Tool(
        name="ComparePapers",
        func=compare_papers,
        description="Compare two papers by approximate title. Input like: 'Paper A and Paper B' or 'Paper A vs. Paper B'."
    ),
]


# ---------------------------
# LLM & Agent setup
# ---------------------------
def build_agent():
    api_key = st.secrets["general"].get("openai_api_key", None)
    if not api_key:
        raise RuntimeError("OpenAI API key missing in secrets.")

    # Chat LLM (fast & inexpensive; adjust as you like)
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.2,
    )

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )

    # ReAct-style prompt (no hub dependency)
    system_prompt = (
        "You are a helpful research assistant that uses the ReAct pattern. "
        "You have two tools: SearchPapers and ComparePapers. "
        "When users ask about topics, first consider if SearchPapers can help; "
        "when they ask to compare two specific titles, consider ComparePapers. "
        "Always keep responses concise, cite the tool outputs where relevant, and "
        "maintain context from chat_history."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("assistant", "Let's think step by step."),
        ]
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=False,
    )
    return executor


if st.session_state.lab6_agent is None:
    try:
        with st.spinner("Spinning up the ReAct agent..."):
            st.session_state.lab6_agent = build_agent()
            time.sleep(0.2)
    except Exception as e:
        st.error(f"Agent Error: {e}")
        st.stop()

def main1():
    # ---------------------------
    # Chat interface
    # ---------------------------
    st.divider()
    st.subheader("Chat")

    with st.container():
        # Render existing history
        for msg in st.session_state.lab6_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("💬 Ask about research papers...")
        if user_input:
            st.session_state.lab6_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.lab6_agent.invoke(
                            {
                                "input": user_input,
                                "chat_history": [
                                    (m["role"], m["content"]) for m in st.session_state.lab6_messages
                                ],
                            }
                        )
                        output = response.get("output", str(response))
                    except Exception as e:
                        output = f"⚠️ Agent failed: {e}"

                st.markdown(output)
                st.session_state.lab6_messages.append({"role": "assistant", "content": output})

def run():
    main1()
if __name__ == "__main__":
    run()