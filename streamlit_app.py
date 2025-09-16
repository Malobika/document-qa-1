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

    sys_text = ""
    content = []
    for m in messages:
        if m["role"] == "system":
            sys_text += m["content"] + "\n"
        else:
            content.append({"role": m["role"], "content": m["content"]})

    model_id = "claude-3-haiku-20240307" if "haiku" in model_name else "claude-3-5-sonnet-20240620"
    with client.messages.stream(
        model=model_id,
        system=sys_text if sys_text else None,
        max_tokens=1000,
        temperature=0.3,
        messages=content,
    ) as stream:
        for event in stream:
            if event.type == "content.delta" and hasattr(event, "delta"):
                yield event.delta
            elif event.type == "message_stop":
                break


def stream_cohere(messages, model_name):
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key or not HAS_COHERE:
        yield "[Cohere] Missing COHERE_API_KEY or package."
        return
    co = cohere.Client(api_key)

    # Flatten messages into a single prompt (Cohere currently expects text input)
    sys_text = ""
    convo = []
    for m in messages:
        if m["role"] == "system":
            sys_text += m["content"] + "\n"
        else:
            convo.append(f"{m['role']}: {m['content']}")
    prompt = sys_text + "\n".join(convo)

    model_id = "command-light" if "light" in model_name else "command-r"
    resp = co.chat_stream(model=model_id, message=prompt)

    for event in resp:
        if event.event_type == "text-generation":
            yield event.text


# ---- Compose context (URLs + conversation) ----
def answer_with_context(question: str, memory_mode: str):
    url_texts = []
    if url1:
        url_texts.append(fetch_url_text(url1))
    if url2:
        url_texts.append(fetch_url_text(url2))

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
if st.session_state.messages:
    for m in st.session_state.messages:
        with st.chat_message(m["role"] if m["role"] in ("user", "assistant") else "assistant"):
            st.markdown(m["content"][:4000] + ("..." if len(m["content"]) > 4000 else ""))

user_q = st.chat_input("Ask a question based on the two URLs (if provided).")
if user_q:
    with st.chat_message("assistant"):
        stream_gen = answer_with_context(user_q, memory_mode)
        full = st.write_stream(stream_gen)

    st.session_state.messages.append({"role": "assistant", "content": full})
