import time
import datetime
from typing import List, Dict, Any, Iterable

import streamlit as st
from dotenv import load_dotenv

from rag_ui.config import Settings, validate_settings
from rag_ui.azure_project import build_azure_context
from rag_ui.retrieval import hybrid_search_sdk


st.set_page_config(page_title="Azure RAG Assistant", layout="wide")

HISTORY_LENGTH = 10
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)
DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

SUGGESTIONS = {
    ":blue[:material/local_library:] Who is Harry Potter?": "Who is Harry Potter?",
    ":green[:material/search:] Summarize chapter one": "Summarize chapter one in one paragraph.",
    ":orange[:material/help:] What is Hogwarts?": "What is Hogwarts?",
    ":violet[:material/bolt:] What is Quidditch?": "What is Quidditch?",
    ":red[:material/description:] Describe Harry in one sentence": "Describe Harry Potter in one sentence using the book text.",
}

SYSTEM_PROMPT = (
    "You must answer ONLY using the provided SOURCES. "
    "If the answer is not in the sources, say \"I don't know\". "
    "Return exactly ONE concise paragraph. "
    "No bullet points, no numbering, no repetition."
)

FINAL_PROMPT = """
<sources>
{ctx}
</sources>

Question: {q}

Answer ONLY using the sources above. If missing, say "I don't know".
""".strip()


@st.cache_resource
def get_settings() -> Settings:
    load_dotenv(override=True)
    s = Settings()
    validate_settings(s)
    return s


@st.cache_resource
def get_ctx():
    s = get_settings()
    return build_azure_context(s)


@st.cache_resource
def get_openai_client():
    s = get_settings()
    ctx = get_ctx()
    return ctx.project_client.get_openai_client(api_version=s.openai_api_version)


def format_hits(docs: List[Dict[str, Any]], max_chunks: int = 6, max_chars_per_chunk: int = 1200) -> str:
    blocks = []
    for i, d in enumerate(docs[:max_chunks], start=1):
        title = (d.get("title") or "unknown").strip()
        chunk = (d.get("chunk") or "").strip()[:max_chars_per_chunk]
        blocks.append(f"[{i}] (source={title})\n{chunk}".strip())
    return "\n\n".join(blocks)


def build_full_prompt(question: str, docs: List[Dict[str, Any]], max_chunks: int) -> str:
    ctx = format_hits(docs, max_chunks=max_chunks)
    return FINAL_PROMPT.format(q=question, ctx=ctx)


def stream_llm_answer(prompt: str) -> Iterable[str]:
    s = get_settings()
    client = get_openai_client()
    try:
        stream = client.chat.completions.create(
            model=s.lm_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            stream=True,
        )
        for event in stream:
            if event.choices and event.choices[0].delta:
                token = event.choices[0].delta.content or ""
                if token:
                    yield token
    except Exception:
        completion = client.chat.completions.create(
            model=s.lm_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            stream=False,
        )
        yield (completion.choices[0].message.content or "").strip()


def retrieve_hybrid(query: str, k_final: int, knn: int) -> List[Dict[str, Any]]:
    ctx = get_ctx()
    s = get_settings()
    return hybrid_search_sdk(
        ctx,
        s,
        query=query,
        top_k_vec=knn,
        top=k_final,
        select_fields=["title", "chunk"],
    )


def clear_conversation():
    st.session_state.messages = []
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)


if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)
if "messages" not in st.session_state:
    st.session_state.messages = []


s = get_settings()

st.markdown(
    f"""
<div style="display:flex; align-items:center; gap:0.6rem; padding:0.6rem 0.8rem; border:1px solid rgba(15,23,42,0.08); border-radius:14px; background:#ffffff;">
  <div style="font-size:2rem;">🤖</div>
  <div>
    <div style="font-size:1.4rem; font-weight:700; color:#0f172a;">{s.lm_name}</div>
    <div style="font-size:0.9rem; opacity:0.7; color:#334155;">
      Hybrid retrieval • Azure AI Search
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

title_row = st.container(horizontal=True, vertical_alignment="bottom")
with title_row:
    st.title("Azure RAG Assistant", anchor=False)
with title_row:
    st.button("Restart", icon=":material/refresh:", on_click=clear_conversation)

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)
user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)
user_first_interaction = user_just_asked_initial_question or user_just_clicked_suggestion
has_message_history = len(st.session_state.messages) > 0

if not user_first_interaction and not has_message_history:
    with st.container():
        st.chat_input("Fai una domanda...", key="initial_question")
        st.pills(
            label="Esempi",
            label_visibility="collapsed",
            options=list(SUGGESTIONS.keys()),
            key="selected_suggestion",
        )
    st.stop()

user_message = st.chat_input("Fai una domanda di follow-up...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

with st.sidebar:
    st.header("Settings")

    k_options = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
    knn_options = [2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]

    k_final = st.selectbox("Number of output chunks", k_options, index=k_options.index(8))
    knn = st.selectbox("kNN (dense)", knn_options, index=knn_options.index(15))

    st.divider()
    st.subheader("Examples")
    for label, q in SUGGESTIONS.items():
        if st.button(label, use_container_width=True):
            st.session_state.initial_question = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_message:
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            now = datetime.datetime.now()
            if now - st.session_state.prev_question_timestamp < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep(0.2)
            st.session_state.prev_question_timestamp = now

            hits = retrieve_hybrid(user_message, k_final=k_final, knn=knn)
            prompt = build_full_prompt(user_message, hits, max_chunks=k_final)

            if DEBUG_MODE:
                st.subheader("DEBUG prompt")
                st.code(prompt)

        with st.spinner("Thinking..."):
            placeholder = st.empty()
            acc = ""
            for tok in stream_llm_answer(prompt):
                acc += tok
                placeholder.markdown(acc)

        with st.expander("📚 Used Sources"):
            if not hits:
                st.caption("No sources retrieved.")
            for h in hits:
                st.markdown(f"**{h.get('title','unknown')}**")
                st.markdown(h.get("chunk", ""))
                st.divider()

    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": acc})
    st.session_state.messages = st.session_state.messages[-2 * HISTORY_LENGTH :]
