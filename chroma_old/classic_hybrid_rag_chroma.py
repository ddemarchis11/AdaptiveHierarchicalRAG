from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, Dict, List, Sequence
from textwrap import shorten

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from hybrid_retriever import HybridRetriever
from llm_stub import get_llm_client

from config import DEFAULT_CONFIG, BaselineHybridBM25Config, PipelineConfig 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from textwrap import shorten

response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You answer questions using the provided context only.\n"
                "Rules:\n"
                "- Use ONLY the context below; do not add outside knowledge.\n"
                "- If the context does not fully support an answer, say so clearly.\n"
                "- Answer in at most 3 sentences."
            ),
        ),
        (
            "human",
            (
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
            ),
        ),
    ]
)

def _doc_id(d: Any, i: int) -> str:
    if isinstance(d, dict):
        return str(
            d.get("id")
            or d.get("doc_id")
            or d.get("source")
            or f"rank_{i}"
        )
    meta = getattr(d, "metadata", {}) or {}
    return str(meta.get("id") or meta.get("source") or f"rank_{i}")


def _doc_text(d: Any) -> str:
    if isinstance(d, dict):
        return str(
            d.get("text")
            or d.get("content")
            or d.get("page_content")
            or ""
        )
    return str(getattr(d, "page_content", "") or "")


def _normalize(docs: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {"id": _doc_id(d, i), "text": _doc_text(d), "position": d["metadata"]["position"]}
        for i, d in enumerate(docs, start=1)
    ]

def _hybrid_rrf_search(
    hybrid: HybridRetriever,
    query: str,
    k_dense: int,
    k_sparse: int,
    rrf_k: int,
) -> List[Dict[str, Any]]:

    docs = hybrid.retrieve(
        query=query,
        k_dense=k_dense,
        k_sparse=k_sparse,
        k_final=rrf_k,
    )
    return _normalize(docs)


def get_top_paragraphs(
    question: str,
    hybrid: HybridRetriever,
    cfg: BaselineHybridBM25Config,
):
    ranking = _hybrid_rrf_search(
        hybrid=hybrid,
        query=question,
        k_dense=cfg.topK_dense,
        k_sparse=cfg.topK_sparse,
        rrf_k=cfg.rrf_k,
    )
    return ranking[: cfg.top_k_final]

def build_baseline_hybrid_bm25_chain(
    hybrid: HybridRetriever,
    llm,
    cfg: BaselineHybridBM25Config = BaselineHybridBM25Config(),
):

    answer_chain = response_prompt | llm | StrOutputParser()

    def dbg(title: str, body: str = ""):
        if not cfg.verbose:
            return
        line = "=" * 70
        print(f"\n{line}\n{title}\n{line}")
        if body:
            print(body)

    def fmt_docs(docs, max_items: int) -> str:
        lines = []
        for i, d in enumerate(docs[:max_items], start=1):
            preview = shorten(
                d["text"].replace("\n", " "),
                width=getattr(cfg, "preview_chars", 240),
                placeholder="…",
            )
            prior = d.get("prior_rrf", None)
            prior_txt = f" | prior={prior:.4f}" if prior is not None else ""
            lines.append(f"{i:02d}. [{d['id']}] {preview}{prior_txt}")
        return "\n".join(lines)

    def _pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]

        dbg("1) QUERY ORIGINALE (q0)", question)

        top_docs = get_top_paragraphs(question, hybrid, cfg)
        dbg(
            f"2) TOP PARAGRAPHS (top_k_final={cfg.top_k_final})",
            fmt_docs(top_docs, cfg.top_k_final),
        )

        context = "\n\n".join(
            f"{d['text']}" for d in top_docs
        )

        answer = answer_chain.invoke(
            {"question": question, "context": context}
        )

        return {
            "question": question,
            "answer": answer,       
            "documents": top_docs,  
            "context": context,     
        }

    return RunnableLambda(_pipeline)

def run(question: str, cfg: PipelineConfig = DEFAULT_CONFIG):
    import chromadb
    from hybrid_retriever import HybridRetriever
    from sentence_transformers import SentenceTransformer

    def get_hybrid_retriever(config: PipelineConfig = cfg) -> HybridRetriever:
        chroma_cfg = config.chroma
        hybrid_cfg = config.hybrid

        client_chroma = chromadb.PersistentClient(
            path=chroma_cfg.persist_directory
        )

        col = client_chroma.get_collection(
            chroma_cfg.collection_name_chunks,
        )
        
        model = SentenceTransformer(cfg.embeddings.model_name)

        return HybridRetriever(
            chroma_collection=col,
            rrf_k=hybrid_cfg.rrf_k,
            model=model,
            w_dense=hybrid_cfg.w_dense,
            w_sparse=hybrid_cfg.w_sparse,
        )

    hybrid = get_hybrid_retriever(cfg)
    llm = get_llm_client(cfg)

    baseline_cfg = cfg.retrieval
    rag_baseline = build_baseline_hybrid_bm25_chain(hybrid, llm, baseline_cfg)

    result = rag_baseline.invoke({
        "question": question
    })
    print("Answer:", result["answer"])
    return result

if __name__ == "__main__": run()