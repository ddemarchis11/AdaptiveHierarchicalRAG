import datetime
from dataclasses import dataclass
from typing import Any, Dict, List

import streamlit as st
from neo4j import GraphDatabase

from config import DEFAULT_CONFIG, PipelineConfig
from llm_stub import get_llm_client
from classic_hybrid_rag import build_baseline_chain
from cot_rag import (
    build_cot_rag_chain,
    CoTRAGConfig,
    ClassicHybridBackend,
    GraphRAGBackend,
    build_es_retriever_from_cfg,
)

from ui_utils.history import safe_sleep_for_rate_limit, trim_history, clear_conversation
from ui_utils.docs import normalize_docs_for_ui, render_used_sources
from ui_utils.upload_index import render_upload_and_index_box

st.set_page_config(page_title="RAG Assistant", layout="wide")

HISTORY_LENGTH = 12
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)
DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"


@dataclass
class RunResult:
    question: str
    answer: str
    documents: List[Dict[str, Any]]
    debug: Dict[str, Any]


@st.cache_resource
def get_cfg() -> PipelineConfig:
    return DEFAULT_CONFIG


@st.cache_resource
def get_llm(cfg: PipelineConfig):
    return get_llm_client(cfg)


@st.cache_resource
def get_es_retriever(cfg: PipelineConfig):
    return build_es_retriever_from_cfg(cfg)


@st.cache_resource
def get_neo4j_driver(cfg: PipelineConfig):
    return GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))


@st.cache_resource
def get_chain_naive(cfg: PipelineConfig):
    llm = get_llm(cfg)
    chain = build_baseline_chain(cfg, llm)
    return chain


@st.cache_resource
def get_chain_cot(cfg: PipelineConfig, cot_cfg: CoTRAGConfig):
    llm = get_llm(cfg)
    es_ret = get_es_retriever(cfg)
    backend = ClassicHybridBackend(es_ret, cot_cfg)
    return build_cot_rag_chain(backend, llm, cot_cfg)


@st.cache_resource
def get_chain_graph(cfg: PipelineConfig, cot_cfg: CoTRAGConfig):
    llm = get_llm(cfg)
    es_ret = get_es_retriever(cfg)
    driver = get_neo4j_driver(cfg)
    backend = GraphRAGBackend(driver, es_ret, cfg, cot_cfg)
    return build_cot_rag_chain(backend, llm, cot_cfg)


def run_naive(question: str, k: int, cfg: PipelineConfig) -> RunResult:
    chain = get_chain_naive(cfg)
    payload = {"question": question, "k": int(k)}
    res = chain.invoke(payload)
    docs = normalize_docs_for_ui(res.get("documents", []))
    dbg = {"mode": "naive", "k": k, "index": cfg.es.index_name}
    if DEBUG_MODE:
        dbg["context_preview"] = (res.get("context", "")[:3000] if res.get("context") else "")
    return RunResult(question=question, answer=res.get("answer", ""), documents=docs, debug=dbg)


def run_cot(question: str, k_final: int, max_steps: int, cfg: PipelineConfig) -> RunResult:
    cot_cfg = CoTRAGConfig(
        max_steps=int(max_steps),
        top_k_final=int(k_final),
        verbose=False,
    )
    chain = get_chain_cot(cfg, cot_cfg)
    res = chain.invoke({"question": question})
    docs = normalize_docs_for_ui(res.get("documents", []))
    dbg = {
        "mode": "cot",
        "index": cfg.es.index_name,
        "scratchpad": res.get("scratchpad", ""),
        "history": res.get("history", []),
        "cot_cfg": cot_cfg.__dict__,
    }
    return RunResult(question=question, answer=res.get("answer", ""), documents=docs, debug=dbg)


def run_graph(question: str, k_final: int, max_steps: int, cfg: PipelineConfig) -> RunResult:
    cot_cfg = CoTRAGConfig(
        max_steps=int(max_steps),
        top_k_final=int(k_final),
        verbose=False,
    )
    chain = get_chain_graph(cfg, cot_cfg)
    res = chain.invoke({"question": question})
    docs = normalize_docs_for_ui(res.get("documents", []))
    dbg = {
        "mode": "graph",
        "index": cfg.es.index_name,
        "scratchpad": res.get("scratchpad", ""),
        "history": res.get("history", []),
        "cot_cfg": cot_cfg.__dict__,
    }
    return RunResult(question=question, answer=res.get("answer", ""), documents=docs, debug=dbg)


cfg = get_cfg()
active_index = st.session_state.get("active_index_name", "general")
cfg.es.index_name = active_index

st.markdown(
    """
<div style="display:flex; align-items:center; gap:0.6rem;">
  <div style="font-size:2rem;">🤖</div>
  <div>
    <div style="font-size:1.6rem; font-weight:700;">LLM Assistant</div>
    <div style="font-size:0.9rem; opacity:0.7;">
      Naive • CoT • Graph
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

title_row = st.container()
with title_row:
    cols = st.columns([1, 0.18])
    with cols[0]:
        st.title("Chat with your Docs", anchor=False)

with cols[1]:
    st.button(
        "Restart",
        icon=":material/refresh:",
        use_container_width=True,
        on_click=lambda: clear_conversation(),
    )

with st.sidebar:
    st.header("Settings")
    rag_mode = st.selectbox(
        "Approccio RAG",
        options=["Naive", "CoT", "Graph"],
        index=0,
    )
    st.divider()
    k_final = st.slider("Top-k finale (chunks mostrati)", 2, 12, 6, 1)
    max_steps = 3
    if rag_mode == "CoT":
        max_steps = st.slider("Max steps (CoT)", 2, 6, 3, 1)
    if DEBUG_MODE:
        st.divider()
        st.caption("Debug mode attivo (query param: ?debug=true)")
    st.divider()

    render_upload_and_index_box(cfg)

    st.divider()
    st.caption(f"Active index: `{st.session_state.get('active_index_name', 'general')}`")


user_just_asked_initial_question = bool(st.session_state.get("initial_question"))
user_first_interaction = user_just_asked_initial_question
has_message_history = bool(st.session_state.get("messages"))

if not user_first_interaction and not has_message_history:
    st.session_state.messages = []
    with st.container():
        st.chat_input("Fai una domanda...", key="initial_question")
    st.stop()

user_message = st.chat_input("Fai una domanda di follow-up...")

if not user_message and user_just_asked_initial_question:
    user_message = st.session_state.initial_question

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_message:
    user_message = user_message.replace("$", r"\$")
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            safe_sleep_for_rate_limit(MIN_TIME_BETWEEN_REQUESTS)

            cfg.es.index_name = st.session_state.get("active_index_name", "general")

            if rag_mode == "Naive":
                result = run_naive(user_message, k=k_final, cfg=cfg)
            elif rag_mode == "CoT":
                result = run_cot(user_message, k_final=k_final, max_steps=max_steps, cfg=cfg)
            else:
                result = run_graph(user_message, k_final=k_final, max_steps=max_steps, cfg=cfg)

            if DEBUG_MODE:
                st.subheader("DEBUG")
                st.json(result.debug)

        with st.spinner("Thinking..."):
            st.markdown(result.answer)

        with st.expander("📚 Used Sources"):
            render_used_sources(result.documents)

    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": result.answer})
    trim_history(HISTORY_LENGTH)