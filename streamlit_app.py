import os
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Optional token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False

# --- Vendor SDKs ---
# OpenAI
try:
    from openai import OpenAI as OpenAIClient
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# Anthropic
try:
    import anthropic
    HAS_ANTHROPIC = True
except Exception:
    HAS_ANTHROPIC = False

# Cohere
try:
    import cohere
    HAS_COHERE = True
except Exception:
    HAS_COHERE = False


st.set_page_config(page_title="HW3 – Compare 2 URLs with memory", page_icon="🧪", layout="wide")
st.title("HW3 – Compare 2 URLs with memory")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Inputs & Settings")
    url1 = st.text_input("URL #1", placeholder="https://example.com/article-1")
    url2 = st.text_input("URL #2 (optional)", placeholder="https://example.com/article-2")

    vendor = st.selectbox(
        "LLM Vendor",
        ["OpenAI", "Anthropic", "Cohere"],
        help="Pick the provider"
    )

    model = None
    if vendor == "OpenAI":
        model = st.selectbox("Model", ["gpt-4o-mini (cheap)", "gpt-4o (flagship)"])
    elif vendor == "Anthropic":
        model = st.selectbox("Model", ["claude-3-haiku-20240307 (cheap)", "claude-3-5-sonnet-20240620 (flagship)"])
    else:
        # Cohere models
        model = st.selectbox("Model", ["command-light (cheap)", "command-r (flagship)"])

    memory_mode = st.selectbox(
        "Conversation Memory",
        ["Buffer (last 6 questions)", "Conversation Summary", "Token Buffer (~2000 tokens)"],
        help="How should the bot remember the chat?"
    )

    st.caption("Tip: ask things like ‘Summarize climate impacts’, ‘What year did X happen?’, etc.")

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = ""


# ---- Utility: fetch webpage text ----
def fetch_url_text(url: str, timeout=15):
    if not url:
        return ""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:20000]
    except Exception as e:
        return f"(Could not fetch {url}: {e})"


# ---- Utility: approx tokens ----
def approx_token_count(text: str, enc_name="cl100k_base"):
    if not HAS_TIKTOKEN:
        return max(1, len(text) // 4)
    try:
        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---- Build memory context ----
def build_context_messages(memory_mode: str):
    sys_msg = {
        "role": "system",
        "content": ("Explain answers simply, so a 10-year-old can understand. "
                    "Be clear, friendly, and use short sentences when possible.")
    }

    msgs = st.session_state.messages.copy()

    if memory_mode == "Buffer (last 6 questions)":
        kept = []
        user_turns = 0
        for m in reversed(msgs):
            kept.append(m)
            if m["role"] == "user":
                user_turns += 1
                if user_turns >= 6:
                    break
        msgs = list(reversed(kept))

    elif memory_mode == "Conversation Summary":
        if st.session_state.summary.strip():
            msgs = [{"role": "system", "content": f"Conversation summary:\n{st.session_state.summary}"}] + msgs

    elif memory_mode == "Token Buffer (~2000 tokens)":
        joined = []
        for m in msgs:
            joined.append(f"{m['role'].upper()}: {m['content']}")
        text = "\n".join(joined)
        if approx_token_count(text) > 2000:
            parts = text.split("\n")
            while parts and approx_token_count("\n".join(parts)) > 2000:
                parts.pop(0)
            text = "\n".join(parts)
            msgs = [{"role": "system", "content": "Recent conversation context (trimmed):\n" + text}]

    msgs = [sys_msg] + msgs
    return msgs


# ---- LLM streaming wrappers ----
def stream_openai(messages, model_name):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not HAS_OPENAI:
        yield "[OpenAI] Missing API key or package."
        return
    client = OpenAIClient(api_key=api_key)
    model_id = "gpt-4o-mini" if "mini" in model_name else "gpt-4o"
    stream = client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        temperature=0.3,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content

def stream_anthropic(messages, model_name):
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key or not HAS_ANTHROPIC:
        yield "[Anthropic] Missing API key or package."
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Separate system messages from conversation messages
    system_messages = []
    conversation_messages = []
    
    for m in messages:
        if m["role"] == "system":
            system_messages.append(m["content"])
        else:
            conversation_messages.append(m)
    
    # Join system messages
    system_text = "\n".join(system_messages) if system_messages else None
    
    # Format messages for Anthropic API
    formatted_messages = []
    for m in conversation_messages:
        formatted_messages.append({
            "role": m["role"],
            "content": m["content"]
        })
    
    # Updated model names with current identifiers
    if "haiku" in model_name.lower():
        model_id = "claude-3-haiku-20240307"
    elif "sonnet" in model_name.lower():
        # Use the current Claude Sonnet 4 model
        model_id = "claude-sonnet-4-20250514"
    elif "opus" in model_name.lower():
        # Check if Claude 4 Opus is available, otherwise fall back to Claude 3
        model_id = "claude-3-opus-20240229"  # Update this when Claude 4 Opus is released
    else:
        # Default to Claude Sonnet 4
        model_id = "claude-sonnet-4-20250514"
    
    try:
        with client.messages.stream(
            model=model_id,
            max_tokens=4000,
            temperature=0.3,
            system=system_text,
            messages=formatted_messages
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        yield event.delta.text
                elif event.type == "message_stop":
                    break
    except Exception as e:
        yield f"[Anthropic] Error: {str(e)}"
      

  


def stream_cohere(messages, model_name):
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key or not HAS_COHERE:
        yield "[Cohere] Missing COHERE_API_KEY or package."
        return
    co = cohere.Client(api_key)

    # Convert to Cohere message format
    cohere_msgs = []
    for m in messages:
        if m["role"] in ("system", "user", "assistant"):
            cohere_msgs.append({"role": m["role"], "content": m["content"]})

    model_id = "command-light" if "light" in model_name else "command-r"

    try:
        resp = co.chat_stream(model=model_id, messages=cohere_msgs)
        for event in resp:
            if event.event_type == "text-generation":
                yield event.text
            elif event.event_type == "stream-end":
                break
    except Exception as e:
        yield f"[Cohere Error] {str(e)}"

# ---- Compose context (URLs + conversation) ----
def answer_with_context(question: str, memory_mode: str):
    url_texts = []
    if "url1_text" in st.session_state and st.session_state.url1_text:
        url_texts.append(st.session_state.url1_text)
    if "url2_text" in st.session_state and st.session_state.url2_text:
        url_texts.append(st.session_state.url2_text)

    if url_texts:
        context_blob = "\n\n".join(
            [f"Source {i+1}:\n{t}" for i, t in enumerate(url_texts)]
        )
        st.session_state.messages.append({
            "role": "system",
            "content": ("Use the following webpage content as factual grounding. "
                        "If a fact isn't in the sources or conversation, say you’re not sure.\n\n" + context_blob[:30000])
        })

    st.session_state.messages.append({"role": "user", "content": question})
    msgs = build_context_messages(memory_mode)

    if vendor == "OpenAI":
        return stream_openai(msgs, model)
    elif vendor == "Anthropic":
        return stream_anthropic(msgs, model)
    else:
        return stream_cohere(msgs, model)


# ---- Chat UI ----
# ---- Chat UI ----
if st.session_state.messages:
    for m in st.session_state.messages:
        with st.chat_message(m["role"] if m["role"] in ("user", "assistant") else "assistant"):
            st.markdown(m["content"][:4000] + ("..." if len(m["content"]) > 4000 else ""))

user_q = st.chat_input("Ask a question based on the two URLs (if provided).")

if user_q:
    # First parse and validate URLs
    parsed1 = fetch_url_text(url1) if url1 else ""
    parsed2 = fetch_url_text(url2) if url2 else ""

    if not parsed1 and not parsed2:
        st.error("❌ Could not parse either URL. Please check the links and try again.")
    else:
        # Store parsed text into session so it doesn't refetch every time
        st.session_state.url1_text = parsed1
        st.session_state.url2_text = parsed2

        with st.chat_message("assistant"):
            stream_gen = answer_with_context(user_q, memory_mode)
            full = st.write_stream(stream_gen)

        st.session_state.messages.append({"role": "assistant", "content": full})
