import os
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Optional (better token counting for the 2,000 token buffer)
try:
    import tiktoken
    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False

# --- Vendor SDKs (import lazily later if keys exist) ---
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

# Google Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False


st.set_page_config(page_title="HW3 – Compare 2 URLs with memory", page_icon="🧪", layout="wide")
st.title("HW3 – Compare 2 URLs with memory")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Inputs & Settings")
    url1 = st.text_input("URL #1", placeholder="https://example.com/article-1")
    url2 = st.text_input("URL #2 (optional)", placeholder="https://example.com/article-2")

    vendor = st.selectbox(
        "LLM Vendor",
        ["OpenAI", "Anthropic", "Google (Gemini)"],
        help="Pick the provider"
    )

    model = None
    if vendor == "OpenAI":
        model = st.selectbox("Model", ["gpt-4o-mini (cheap)", "gpt-4o (flagship)"])
    elif vendor == "Anthropic":
        model = st.selectbox("Model", ["claude-3-haiku-20240307 (cheap)", "claude-3-5-sonnet-20240620 (flagship)"])
    else:
        model = st.selectbox("Model", ["gemini-1.5-flash (cheap)", "gemini-1.5-pro (flagship)"])

    memory_mode = st.selectbox(
        "Conversation Memory",
        ["Buffer (last 6 questions)", "Conversation Summary", "Token Buffer (~2000 tokens)"],
        help="How should the bot remember the chat?"
    )

    st.caption("Tip: ask things like ‘Summarize climate impacts’, ‘What year did X happen?’, etc.")

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"/"assistant", "content": "..."}
if "summary" not in st.session_state:
    st.session_state.summary = ""   # running summary for Conversation Summary mode

# ---- Utility: fetch & clean webpage text ----
def fetch_url_text(url: str, timeout=15):
    if not url:
        return ""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style/nav/footer
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        # Light truncate to keep context manageable
        return text[:20000]
    except Exception as e:
        return f"(Could not fetch {url}: {e})"

# ---- Utility: token/char trimming for memory ----
def approx_token_count(text: str, enc_name="cl100k_base"):
    if not HAS_TIKTOKEN:
        # crude approx ~ 4 chars per token
        return max(1, len(text) // 4)
    try:
        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

def build_context_messages(memory_mode: str):
    """
    Returns a list of messages (role, content) to send to the model,
    based on the chosen memory strategy.
    """
    sys_msg = {
        "role": "system",
        "content": ("Explain answers simply, so a 10-year-old can understand. "
                    "Be clear, friendly, and use short sentences when possible.")
    }

    # Base conversation
    msgs = st.session_state.messages.copy()

    if memory_mode == "Buffer (last 6 questions)":
        # Keep last 6 user turns with their assistant replies (roughly)
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
        # Prepend a summary if we have one
        if st.session_state.summary.strip():
            msgs = [{"role": "system", "content": f"Conversation summary so far:\n{st.session_state.summary}"}] + msgs

    elif memory_mode == "Token Buffer (~2000 tokens)":
        # Trim to ~2000 tokens
        # Join, then trim from the left (oldest) to fit the budget
        joined = ""
        joined_list = []
        for m in msgs:
            joined_list.append(f"{m['role'].upper()}: {m['content']}")
        # keep the end of the conversation
        text = "\n".join(joined_list)
        if approx_token_count(text) > 2000:
            # chop from the left until roughly under 2000 tokens
            parts = text.split("\n")
            while parts and approx_token_count("\n".join(parts)) > 2000:
                parts.pop(0)
            text = "\n".join(parts)
            # Rebuild as a single user preface message
            msgs = [{"role": "system", "content": "Recent conversation context (trimmed):\n" + text}]

    # Always start with kid-friendly system message
    msgs = [sys_msg] + msgs
    return msgs

# ---- LLM streaming wrappers ----
def stream_openai(messages, model_name):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield "[OpenAI] Missing OPENAI_API_KEY."
        return
    if not HAS_OPENAI:
        yield "[OpenAI] openai package not available."
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield "[Anthropic] Missing ANTHROPIC_API_KEY."
        return
    if not HAS_ANTHROPIC:
        yield "[Anthropic] anthropic package not available."
        return

    client = anthropic.Anthropic(api_key=api_key)
    # Convert OpenAI-style messages to Anthropic's
    sys_text = ""
    content = []
    for m in messages:
        role = m["role"]
        if role == "system":
            sys_text += m["content"] + "\n"
        elif role in ("user", "assistant"):
            content.append({"role": role, "content": m["content"]})

    model_id = "claude-3-haiku-20240307" if "haiku" in model_name else "claude-3-5-sonnet-20240620"
    with client.messages.stream(
        model=model_id,
        system=sys_text if sys_text else None,
        max_tokens=1000,
        temperature=0.3,
        messages=content,
    ) as stream:
        for event in stream:
            if event.type == "content.delta" and hasattr(event, "delta") and event.delta:
                yield event.delta
            elif event.type == "message_stop":
                break

def stream_gemini(messages, model_name):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield "[Gemini] Missing GEMINI_API_KEY or GOOGLE_API_KEY."
        return
    if not HAS_GEMINI:
        yield "[Gemini] google-generativeai package not available."
        return

    genai.configure(api_key=api_key)
    model_id = "gemini-1.5-flash" if "flash" in model_name else "gemini-1.5-pro"

    # Convert messages to a single prompt (Gemini can take role-based too, but simple path)
    sys = []
    convo = []
    for m in messages:
        if m["role"] == "system":
            sys.append(m["content"])
        else:
            convo.append(f"{m['role']}: {m['content']}")
    prompt = ""
    if sys:
        prompt += "\n".join(sys) + "\n\n"
    prompt += "\n".join(convo)

    model = genai.GenerativeModel(model_id)
    resp = model.generate_content(prompt, stream=True)
    for chunk in resp:
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text

# ---- Compose context (URLs + conversation) and stream answer ----
def answer_with_context(question: str, memory_mode: str):
    # Fetch URL texts (if any)
    url_texts = []
    if url1:
        url_texts.append(fetch_url_text(url1))
    if url2:
        url_texts.append(fetch_url_text(url2))

    context_blob = ""
    if url_texts:
        context_blob = "\n\n".join(
            [f"Source {i+1} ({u}):\n{t}" for i, (u, t) in enumerate([(url1, url_texts[0])] + ([(url2, url_texts[1])] if len(url_texts) > 1 else []))]
        )
        # place as a system preface to steer grounding
        st.session_state.messages.append({
            "role": "system",
            "content": ("Use the following webpage content as factual grounding when answering. "
                        "If a fact isn't in the sources or conversation, say you’re not sure.\n\n" + context_blob[:30000])
        })

    # Add the user's new question
    st.session_state.messages.append({"role": "user", "content": question})

    # Build messages with memory strategy
    msgs = build_context_messages(memory_mode)

    # Stream via chosen vendor
    if vendor == "OpenAI":
        return stream_openai(msgs, model)
    elif vendor == "Anthropic":
        return stream_anthropic(msgs, model)
    else:
        return stream_gemini(msgs, model)

# ---- Update conversation summary (if chosen) ----
def update_summary(latest_user: str, latest_assistant: str):
    if memory_mode != "Conversation Summary":
        return
    # Use a tiny prompt to compress the ongoing dialogue
    compress_msgs = [
        {"role": "system", "content": "You compress chats into a short, neutral running summary."},
        {"role": "user", "content": f"Current summary:\n{st.session_state.summary}\n\nNew turn:\nUser: {latest_user}\nAssistant: {latest_assistant}\n\nUpdate the summary in ~3-5 sentences."}
    ]
    # Try using the same vendor to summarize
    generator = None
    if vendor == "OpenAI":
        generator = stream_openai(compress_msgs, "gpt-4o-mini (cheap)")
    elif vendor == "Anthropic":
        generator = stream_anthropic(compress_msgs, "claude-3-haiku-20240307 (cheap)")
    else:
        generator = stream_gemini(compress_msgs, "gemini-1.5-flash (cheap)")

    chunks = []
    for ch in generator:
        chunks.append(ch or "")
    st.session_state.summary = "".join(chunks).strip()

# ---- Chat UI ----
# Show previous chat
if st.session_state.messages:
    for m in st.session_state.messages:
        with st.chat_message(m["role"] if m["role"] in ("user", "assistant") else "assistant"):
            st.markdown(m["content"][:4000] + ("..." if len(m["content"]) > 4000 else ""))

# Prompt
user_q = st.chat_input("Ask a question based on the two URLs (if provided).")
if user_q:
    # Stream the answer
    with st.chat_message("assistant"):
        stream_gen = answer_with_context(user_q, memory_mode)
        # Stream to UI
        full = st.write_stream(stream_gen)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": full})

    # Maintain memory variants
    if memory_mode == "Buffer (last 6 questions)":
        # Drop oldest user turns beyond 6 (keep assistant responses tied loosely)
        user_count = 0
        i = len(st.session_state.messages) - 1
        # count from end, find 6th user
        for j in range(len(st.session_state.messages) - 1, -1, -1):
            if st.session_state.messages[j]["role"] == "user":
                user_count += 1
                if user_count == 6:
                    cutoff_idx = j
                    # drop everything before cutoff_idx except system messages
                    st.session_state.messages = [
                        m for k, m in enumerate(st.session_state.messages)
                        if (k >= cutoff_idx or m["role"] == "system")
                    ]
                    break

    elif memory_mode == "Conversation Summary":
        update_summary(user_q, full)

    elif memory_mode == "Token Buffer (~2000 tokens)":
        # Nothing persistent here; trimming happens in build_context_messages
        pass

# ---- Deployment status (public but repo/app set to private) ----
public_url = os.getenv("PUBLIC_APP_URL", "").strip()
with st.sidebar:
    st.divider()
    st.subheader("Deployment")
    if public_url:
        st.markdown(f"**Public App URL:** [{public_url}]({public_url})")
        st.caption("Open this link to verify the page loads and streams answers.")
    else:
        st.caption("Set env var `PUBLIC_APP_URL` to show your public URL here for a quick check.")
