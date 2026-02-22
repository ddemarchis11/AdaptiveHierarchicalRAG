from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Optional
from textwrap import shorten
import json
import inspect

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from hybrid_retriever import HybridRetriever

@dataclass
class CoTRAGConfig:
    n_stepback_queries: int = 1
    k_per_step_dense: int = 12
    k_per_step_sparse: int = 12
    rrf_k_per_step: int = 60
    k_pool_per_step: int = 30
    use_cross_encoder: bool = True
    cross_encoder_model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k_ce_rerank: int = 20
    top_k_step_context: int = 5
    max_steps: int = 4
    stop_if_enough: bool = True
    allow_adaptive_followup: bool = True
    dedup_across_steps: bool = True
    max_evidence_chars: int = 1200
    top_k_final: int = 6
    final_global_rerank_pool: int = 60
    verbose: bool = True
    preview_chars: int = 180

def _doc_id(d: Any, i: int) -> str:
    if isinstance(d, dict):
        return str(d.get("id") or d.get("doc_id") or d.get("source") or f"rank_{i}")
    meta = getattr(d, "metadata", {}) or {}
    return str(meta.get("id") or meta.get("source") or f"rank_{i}")

def _doc_text(d: Any) -> str:
    if isinstance(d, dict):
        return str(d.get("text") or d.get("content") or d.get("page_content") or "")
    return str(getattr(d, "page_content", "") or "")

def _normalize(docs: Sequence[Any]) -> List[Dict[str, Any]]:
    return [{"id": _doc_id(d, i), "text": _doc_text(d), "raw": d}
            for i, d in enumerate(docs, start=1)]

def _rrf_fallback(rankings: Sequence[Sequence[Dict[str, Any]]], k0: int = 60) -> List[Dict[str, Any]]:
    scores = defaultdict(float)
    by_id: Dict[str, Dict[str, Any]] = {}
    for docs in rankings:
        for rank, d in enumerate(docs, start=1):
            by_id[d["id"]] = d
            scores[d["id"]] += 1.0 / (k0 + rank)
    fused = sorted(by_id.values(), key=lambda d: scores[d["id"]], reverse=True)
    for d in fused:
        d["prior_rrf"] = scores[d["id"]]
    return fused

def _hybrid_rrf_search(
    hybrid: HybridRetriever,
    query: str,
    k_dense: int,
    k_sparse: int,
    rrf_k: int
) -> List[Dict[str, Any]]:
    candidates = ["search", "hybrid_search", "rrf_search", "retrieve", "get_relevant_documents"]
    for name in candidates:
        if hasattr(hybrid, name):
            fn = getattr(hybrid, name)
            try:
                sig = inspect.signature(fn)
                kwargs = {}
                for p in sig.parameters.values():
                    if p.name in ("q", "query", "question", "text"):
                        kwargs[p.name] = query
                    elif p.name in ("k_dense", "dense_k", "topk_dense"):
                        kwargs[p.name] = k_dense
                    elif p.name in ("k_sparse", "sparse_k", "topk_sparse"):
                        kwargs[p.name] = k_sparse
                    elif p.name in ("rrf_k", "k0", "rrf_k0"):
                        kwargs[p.name] = rrf_k
                out = fn(**kwargs)
                return _normalize(out)
            except Exception:
                pass

    dense_ids = hybrid._dense_search(query, k_dense)
    sparse_ids = hybrid._sparse_search(query, k_sparse)
    dense_docs = []
    for cid in dense_ids:
        txt, meta = hybrid._lookup[cid]
        dense_docs.append({"id": cid, "text": txt, "metadata": meta})
    sparse_docs = []
    for cid in sparse_ids:
        txt, meta = hybrid._lookup[cid]
        sparse_docs.append({"id": cid, "text": txt, "metadata": meta})
    dense_norm = _normalize(dense_docs)
    sparse_norm = _normalize(sparse_docs)
    return _rrf_fallback([dense_norm, sparse_norm], k0=rrf_k)

def cross_encoder_rerank(
    q0: str, docs: List[Dict[str, Any]], model_name: str, top_k: int
) -> List[Dict[str, Any]]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return docs
    ce = CrossEncoder(model_name)
    cand = docs[:top_k]
    pairs = [[q0, d["text"]] for d in cand]
    scores = ce.predict(pairs)
    for d, s in zip(cand, scores):
        d["score_ce"] = float(s)
    cand_sorted = sorted(cand, key=lambda d: d["score_ce"], reverse=True)
    rest = docs[top_k:]
    return cand_sorted + rest

examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": json.dumps({
            "stepback_question": "what authorities and actions are typically available to members of a police organization?",
            "intent": "definition / capabilities"
        })
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": json.dumps({
            "stepback_question": "what is Jan Sindel’s personal history and background?",
            "intent": "biographical / background"
        })
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

stepback_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at query abstraction for retrieval.\n"
     "Given a user question, do ONE level of abstraction upward to capture the general concept and intent.\n"
     "Rules:\n"
     "- Do NOT introduce new entities or change the subject.\n"
     "- You may drop low-level details if they are not essential to the general concept.\n"
     "- Keep the topic the same, but broader.\n"
     "Output a JSON object with two fields:\n"
     "1) \"stepback_question\": a single generic question that helps retrieve background/principles.\n"
     "2) \"intent\": a short label of the information need.\n"
     "Return ONLY JSON."),
    few_shot_prompt,
    ("user", "{question}"),
])

def make_stepback_gen(llm):
    def parse_json(s: str) -> Dict[str, str]:
        s = s.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "stepback_question" in obj:
                return {
                    "stepback_question": str(obj["stepback_question"]).strip(),
                    "intent": str(obj.get("intent", "")).strip()
                }
        except Exception:
            pass
        return {"stepback_question": s, "intent": ""}
    return (
        stepback_prompt
        | llm.bind(temperature=0)
        | StrOutputParser()
        | RunnableLambda(parse_json)
    )

def make_planner(llm):
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a retrieval planner for multi-hop QA.\n"
         "Given the original question, a generic step-back question, and intent,\n"
         "produce a short plan of 2-4 steps.\n"
         "Each step must contain:\n"
         "- subquestion: what to find in this step\n"
         "- search_query: a concise search query for retrieval\n"
         "Return ONLY valid JSON with a single top-level key called \"steps\"."),
        ("user", "Question: {question}\nStep-back: {generic}\nIntent: {intent}")
    ])
    def parse_json(s: str):
        try:
            obj = json.loads(s.strip())
            steps = obj.get("steps", [])
            if not isinstance(steps, list) or not steps:
                steps = [{"subquestion": "", "search_query": s.strip()}]
            return {"steps": steps}
        except Exception:
            return {"steps": [{"subquestion": "", "search_query": s.strip()}]}
    return planner_prompt | llm.bind(temperature=0) | StrOutputParser() | RunnableLambda(parse_json)

def make_step_reasoner(llm):
    step_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You reason step-by-step but do NOT reveal private chain-of-thought.\n"
         "Given the question, current step, current scratchpad, and new evidence,\n"
         "update the scratchpad as short bullet facts.\n"
         "Decide if you already have enough to answer.\n"
         "If not, propose ONE next search query.\n"
         "Return ONLY valid JSON with keys: scratchpad_update, enough, missing, next_query."),
        ("user", "Question: {question}\nCurrent step: {step}\nScratchpad:\n{scratchpad}\nEvidence:\n{evidence}")
    ])
    def parse_json(s: str):
        try:
            obj = json.loads(s.strip())
            return {
                "scratchpad_update": str(obj.get("scratchpad_update", "")).strip(),
                "enough": bool(obj.get("enough", False)),
                "missing": str(obj.get("missing", "")).strip(),
                "next_query": str(obj.get("next_query", "")).strip()
            }
        except Exception:
            return {"scratchpad_update": s.strip(), "enough": False, "missing": "", "next_query": ""}
    return step_prompt | llm.bind(temperature=0) | StrOutputParser() | RunnableLambda(parse_json)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided evidence. If evidence is insufficient, say so clearly."),
    ("user", "Question: {question}\n\nEvidence:\n{context}\n\n")
])

def build_cot_rag_chain(hybrid: HybridRetriever, llm, cfg: CoTRAGConfig = CoTRAGConfig()):
    stepback_gen = make_stepback_gen(llm)
    planner = make_planner(llm)
    step_reasoner = make_step_reasoner(llm)

    def dbg(title: str, body: str = ""):
        if not cfg.verbose: return
        print(f"\n{'='*70}\n{title}\n{'='*70}")
        if body: print(body)

    def fmt_docs(docs: List[Dict[str, Any]], max_items: int):
        lines = []
        for i, d in enumerate(docs[:max_items], start=1):
            preview = shorten(d["text"].replace("\n", " "), width=cfg.preview_chars, placeholder="…")
            prior = d.get("prior_rrf", None)
            ces = d.get("score_ce", None)
            prior_txt = f" | prior={prior:.4f}" if prior is not None else ""
            ce_txt = f" | ce={ces:.4f}" if ces is not None else ""
            lines.append(f"{i:02d}. [{d['id']}] {preview}{prior_txt}{ce_txt}")
        return "\n".join(lines)

    def compress_evidence(docs: List[Dict[str, Any]], limit_chars: int):
        text = "\n".join(f"[{d['id']}] {d['text']}" for d in docs)
        return text[:limit_chars]

    def _pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
        q0 = inputs["question"]
        dbg("1) QUERY ORIGINALE (q0)", q0)
        sb_obj = stepback_gen.invoke({"question": q0})
        qg_base = sb_obj["stepback_question"]
        intent = sb_obj.get("intent", "")
        stepbacks = [qg_base]
        if cfg.n_stepback_queries > 1:
            tmp_prompt = ChatPromptTemplate.from_messages([
                ("system", f"Generate {cfg.n_stepback_queries - 1} diverse generic rewrites."),
                ("user", "{question}")
            ])
            extra = (tmp_prompt | llm.bind(temperature=0) | StrOutputParser() | RunnableLambda(lambda s: [x.strip() for x in s.splitlines() if x.strip()])).invoke({"question": qg_base})
            stepbacks += extra
        dbg("2) STEP-BACK", f"qg_base: {qg_base}\nintent: {intent}")
        plan_obj = planner.invoke({"question": q0, "generic": qg_base, "intent": intent})
        steps = plan_obj["steps"][:cfg.max_steps]
        dbg("3) PLAN", json.dumps(steps, indent=2))
        scratchpad, all_evidence, seen_ids = "", [], set()
        for t, step in enumerate(steps, start=1):
            subq = step.get("subquestion", f"step_{t}")
            query = step.get("search_query", q0)
            dbg(f"4.{t}) RETRIEVAL '{subq}' | query='{query}'")
            ranking = _hybrid_rrf_search(hybrid, query, cfg.k_per_step_dense, cfg.k_per_step_sparse, cfg.rrf_k_per_step)
            ranking = ranking[:cfg.k_pool_per_step]
            if cfg.use_cross_encoder:
                ranking = cross_encoder_rerank(q0, ranking, cfg.cross_encoder_model_name, min(cfg.top_k_ce_rerank, len(ranking)))
            dbg(f"4.{t}) RANKING", fmt_docs(ranking, 5))
            step_docs = ranking[:cfg.top_k_step_context]
            if cfg.dedup_across_steps:
                step_docs = [d for d in step_docs if d["id"] not in seen_ids]
                for d in step_docs: seen_ids.add(d["id"])
            all_evidence.extend(step_docs)
            ev_text = compress_evidence(step_docs, cfg.max_evidence_chars)
            upd = step_reasoner.invoke({"question": q0, "step": subq, "scratchpad": scratchpad, "evidence": ev_text})
            if upd["scratchpad_update"]:
                scratchpad = (scratchpad + "\n" + upd["scratchpad_update"]).strip()
            dbg(f"4.{t}) STOP?", f"enough={upd['enough']}")
            if cfg.stop_if_enough and upd["enough"]: break
            if (cfg.allow_adaptive_followup and upd["next_query"] and t == len(steps) and t < cfg.max_steps):
                steps.append({"subquestion": upd["missing"] or f"followup_{t+1}", "search_query": upd["next_query"]})
        if cfg.use_cross_encoder and len(all_evidence) > cfg.final_global_rerank_pool:
            all_evidence = cross_encoder_rerank(q0, all_evidence, cfg.cross_encoder_model_name, cfg.final_global_rerank_pool)
        final_docs = all_evidence[:cfg.top_k_final]
        context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in final_docs)
        return {"question": q0, "context": context}

    return RunnableLambda(_pipeline) | final_prompt | llm | StrOutputParser()

if __name__ == "__main__":
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    from langchain_openai import ChatOpenAI

    def get_hybrid_retriever():
        client_chroma = chromadb.PersistentClient(path="./storage/chroma")
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="thenlper/gte-small")
        col = client_chroma.get_collection("rag_tesi", embedding_function=embedding_fn)
        return HybridRetriever(chroma_collection=col, rrf_k=60, w_dense=1.0, w_sparse=1.0)

    def get_llm_client():
        return ChatOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")

    hybrid = get_hybrid_retriever()
    llm = get_llm_client()
    cfg = CoTRAGConfig(verbose=True)
    rag = build_cot_rag_chain(hybrid, llm, cfg)
    out = rag.invoke({"question": "Your question here"})
    print(out)