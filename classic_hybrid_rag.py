from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, Dict, List, Optional
from textwrap import shorten

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from llm_stub import get_llm_client
from config import DEFAULT_CONFIG, PipelineConfig, BaselineHybridBM25Config

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
        ("human", "Question:\n{question}\n\nContext:\n{context}\n\n"),
    ]
)


class ElasticsearchRetriever:
    def __init__(self, client: Elasticsearch, model: SentenceTransformer, index_name: str):
        self.client = client
        self.model = model
        self.index_name = index_name

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
        body = {
            "size": k,
            "query": {"match": {"text": query}},
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": max(k * 10, 100),
            },
            "_source": ["chunk_id", "text", "position", "doc_id"],
        }
        resp = self.client.search(index=self.index_name, body=body)

        out: List[Dict[str, Any]] = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {}) or {}
            out.append(
                {
                    "id": src.get("chunk_id", hit.get("_id")),
                    "text": src.get("text", ""),
                    "score": float(hit.get("_score", 0.0)),
                    "metadata": {"position": src.get("position"), "doc_id": src.get("doc_id")},
                }
            )
        return out


def build_es_client_from_cfg(cfg: PipelineConfig) -> Elasticsearch:
    return Elasticsearch(
        cfg.es.url,
        basic_auth=(cfg.es.user, cfg.es.password) if getattr(cfg.es, "user", None) else None,
        verify_certs=getattr(cfg.es, "verify_certs", True),
    )


def build_retriever_from_cfg(cfg: PipelineConfig) -> ElasticsearchRetriever:
    model = SentenceTransformer(cfg.embeddings.model_name)
    return ElasticsearchRetriever(
        client=build_es_client_from_cfg(cfg),
        model=model,
        index_name=cfg.es.index_name,
    )


def build_baseline_chain(
    cfg: PipelineConfig,
    llm,
    retriever: Optional[ElasticsearchRetriever] = None,
    retrieval_cfg: Optional[BaselineHybridBM25Config] = None,
):
    retriever = retriever or build_retriever_from_cfg(cfg)
    retrieval_cfg = retrieval_cfg or cfg.retrieval
    answer_chain = response_prompt | llm | StrOutputParser()

    def dbg(title: str, body: str = ""):
        if not getattr(retrieval_cfg, "verbose", False):
            return
        print(f"\n{'='*70}\n{title}\n{'='*70}")
        if body:
            print(body)

    def fmt_docs(docs: List[Dict[str, Any]], max_items: int) -> str:
        width = getattr(retrieval_cfg, "preview_chars", 240)
        lines = []
        for i, d in enumerate(docs[:max_items], start=1):
            preview = shorten(d.get("text", "").replace("\n", " "), width=width, placeholder="…")
            score = float(d.get("score", 0.0))
            lines.append(f"{i:02d}. [{d.get('id')}] {preview} | score={score:.4f}")
        return "\n".join(lines)

    def _pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = str(inputs.get("question", "")).strip()
        if not question:
            raise ValueError("Missing 'question' in inputs")

        k = int(inputs.get("k", retrieval_cfg.top_k_final))

        dbg("1) QUESTION (used as query too)", question)

        docs = retriever.retrieve(question, k=k)
        dbg(f"2) RETRIEVED DOCS (k={k})", fmt_docs(docs, k))

        context = "\n\n".join(d.get("text", "") for d in docs)
        answer = answer_chain.invoke({"question": question, "context": context})

        return {"question": question, "answer": answer, "documents": docs, "context": context}

    return RunnableLambda(_pipeline)


def run(question: str, cfg: PipelineConfig = DEFAULT_CONFIG, k: Optional[int] = None):
    llm = get_llm_client(cfg)
    chain = build_baseline_chain(cfg, llm)
    payload: Dict[str, Any] = {"question": question}
    if k is not None:
        payload["k"] = int(k)
    result = chain.invoke(payload)
    print("\n=== ANSWER ===")
    print(result["answer"])
    return result


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "PLACEHOLDER"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(q, k=k)
