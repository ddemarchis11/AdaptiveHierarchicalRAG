from __future__ import annotations

import json
import inspect
import logging
from dataclasses import dataclass
from collections import defaultdict
from textwrap import shorten
from typing import Any, Dict, List, Sequence, Tuple, Protocol

import chromadb
from chromadb.config import Settings
from neo4j import GraphDatabase, Session

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from sentence_transformers import SentenceTransformer

from config import DEFAULT_CONFIG, PipelineConfig
from hybrid_retriever import HybridRetriever
from graph_rag_utils.community import ensure_gds_graph
from llm_stub import get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class CoTRAGConfig:
    k_per_step_dense: int = 12
    k_per_step_sparse: int = 12
    rrf_k_per_step: int = 60
    k_pool_per_step: int = 30
    top_k_step_context: int = 5
    max_steps: int = 4
    stop_if_enough: bool = True
    dedup_across_steps: bool = True
    max_evidence_chars: int = 1200
    history_window: int = 2
    top_k_final: int = 6
    verbose: bool = True
    preview_chars: int = 180


def _doc_id(d: Any, i: int) -> str:
    if isinstance(d, dict):
        return str(
            d.get("id")
            or d.get("chunk_id")
            or d.get("doc_id")
            or d.get("source")
            or f"rank_{i}"
        )
    meta = getattr(d, "metadata", {}) or {}
    return str(meta.get("id") or meta.get("source") or f"rank_{i}")


def _doc_text(d: Any) -> str:
    if isinstance(d, dict):
        return str(d.get("text") or d.get("content") or d.get("page_content") or "")
    return str(getattr(d, "page_content", "") or "")


def _normalize(docs: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": _doc_id(d, i),
            "text": _doc_text(d),
            "raw": d,
        }
        for i, d in enumerate(docs, start=1)
    ]


def _rrf_fallback(
    rankings: Sequence[Sequence[Dict[str, Any]]], k0: int = 60
) -> List[Dict[str, Any]]:
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
    rrf_k: int,
) -> List[Dict[str, Any]]:
    candidates = [
        "search",
        "hybrid_search",
        "rrf_search",
        "retrieve",
        "get_relevant_documents",
    ]
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


def personalized_pagerank(
    session: Session,
    seed_ids: List[str],
    cfg: PipelineConfig,
) -> List[Tuple[str, float]]:
    if not seed_ids:
        return []

    ensure_gds_graph(session, cfg)
    query = """
    MATCH (c:Chunk)
    WHERE c.chunk_id IN $seed_ids
    WITH collect(id(c)) AS sourceNodes
    CALL gds.pageRank.stream($graph_name, {
        sourceNodes: sourceNodes,
        maxIterations: 30
    })
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).chunk_id AS chunk_id, score
    ORDER BY score DESC
    """
    records = session.run(
        query,
        seed_ids=seed_ids,
        graph_name=cfg.graph.graph_name,
    )
    return [(r["chunk_id"], r["score"]) for r in records]


def fetch_chunk_metadata(
    session: Session,
    chunk_ids: List[str],
) -> Dict[str, dict]:
    if not chunk_ids:
        return {}

    query = """
    MATCH (c:Chunk)
    WHERE c.chunk_id IN $chunk_ids
    RETURN c.chunk_id AS chunk_id,
           c.doc_id AS doc_id,
           c.position AS position,
           c.text AS text,
           c.community_id AS community_id
    """
    result = session.run(query, chunk_ids=chunk_ids)
    meta: Dict[str, dict] = {}
    for record in result:
        meta[record["chunk_id"]] = {
            "doc_id": record["doc_id"],
            "position": record["position"],
            "text": record["text"],
            "community_id": record["community_id"],
        }
    return meta


def rank_communities(
    ppr_scores: List[Tuple[str, float]],
    metadata: Dict[str, dict],
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for chunk_id, score in ppr_scores:
        community_id = metadata.get(chunk_id, {}).get("community_id")
        if community_id is None:
            continue
        scores[community_id] = scores.get(community_id, 0.0) + score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def get_seed_chunks(
    hybrid: HybridRetriever,
    query: str,
    cfg: PipelineConfig,
) -> List[str]:
    results = hybrid.retrieve(
        query=query,
        k_dense=cfg.query_top_k_chunks,
        k_sparse=cfg.query_top_k_chunks,
        k_final=cfg.query_top_k_chunks,
    )
    return [r["id"] for r in results]


def graph_rag_retrieve(
    session: Session,
    hybrid: HybridRetriever,
    query: str,
    cfg: PipelineConfig,
    max_results: int,
) -> List[Dict[str, Any]]:
    seed_ids = get_seed_chunks(hybrid, query, cfg)
    if not seed_ids:
        return []

    ppr_scores = personalized_pagerank(session, seed_ids, cfg)
    if not ppr_scores:
        return []

    chunk_ids = [cid for cid, _ in ppr_scores[: cfg.query_top_k_chunks * 5]]
    meta = fetch_chunk_metadata(session, chunk_ids)
    community_scores = rank_communities(ppr_scores, meta)
    top_cids = [cid for cid, _ in community_scores[: cfg.query_top_k_communities]]

    filtered_chunks: List[Tuple[str, float]] = []
    for chunk_id, score in ppr_scores:
        community_id = meta.get(chunk_id, {}).get("community_id")
        if community_id in top_cids:
            filtered_chunks.append((chunk_id, score))
        if len(filtered_chunks) >= max_results:
            break

    docs: List[Dict[str, Any]] = []
    for chunk_id, score in filtered_chunks:
        m = meta.get(chunk_id)
        if not m:
            continue
        docs.append(
            {
                "id": chunk_id,
                "text": m["text"],
                "metadata": {
                    "doc_id": m["doc_id"],
                    "position": m["position"],
                    "community_id": m["community_id"],
                    "score_ppr": score,
                },
            }
        )
    return docs


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        ...


class ClassicHybridBackend:
    def __init__(self, hybrid: HybridRetriever, cfg_cot: CoTRAGConfig):
        self.hybrid = hybrid
        self.cfg_cot = cfg_cot

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        docs = _hybrid_rrf_search(
            self.hybrid,
            query=query,
            k_dense=self.cfg_cot.k_per_step_dense,
            k_sparse=self.cfg_cot.k_per_step_sparse,
            rrf_k=self.cfg_cot.rrf_k_per_step,
        )
        return docs[:k]


class GraphRAGBackend:
    def __init__(
        self,
        driver,
        hybrid: HybridRetriever,
        graph_cfg: PipelineConfig,
        cot_cfg: CoTRAGConfig,
    ):
        self.driver = driver
        self.hybrid = hybrid
        self.graph_cfg = graph_cfg
        self.cot_cfg = cot_cfg

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.graph_cfg.neo4j.database) as session:
            docs = graph_rag_retrieve(
                session=session,
                hybrid=self.hybrid,
                query=query,
                cfg=self.graph_cfg,
                max_results=k,
            )
        return docs


def make_step_reasoner(llm):
    step_prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            (
                "You are a strategic retrieval controller for multi-hop reasoning.\n"
                "Output ONLY valid JSON. Do not use tools.\n\n"
                "Goal: Identify the 'Bridge Entity' that connects the start concept to the end concept.\n\n"
                "### METHODOLOGY (Multi-Hop Pivot):\n"
                "1. **Analyze Evidence**: Look for Proper Nouns (places, people, organizations) that are associated with the subject of the question.\n"
                "2. **Identify the Bridge**: If Question asks 'Link A to C', and Evidence says 'A is related to B', then B is your likely Bridge. You must now search for the link between B and C.\n"
                "3. **Refine Query**: Do not repeat general searches. Use the specific Bridge Entity found.\n\n"
                "### EXAMPLES of Reasoning Logic (Do not copy, apply the logic):\n\n"
                "**Case 1: Finding a specific feature**\n"
                "- Question: 'What statue is inside the Green Park?'\n"
                "- Evidence found: 'The Green Park contains the famous Rose Garden section.'\n"
                "- BAD Query: 'statue in Green Park' (Repeated failure)\n"
                "- GOOD Query: 'statues in Rose Garden' (Pivoting on the new entity found)\n\n"
                "**Case 2: Linking via relation**\n"
                "- Question: 'Who is the director of the movie starring Actor X?'\n"
                "- Evidence found: 'Actor X is best known for his role in the movie The Great Void.'\n"
                "- BAD Query: 'movies starring Actor X'\n"
                "- GOOD Query: 'director of The Great Void' (Pivoting on the movie title found)\n\n"
                "### CRITICAL RULES for 'next_query':\n"
                "1. **NO REPETITION**: If searching for the target directly failed, do not try again. Change the focus to the container/location/person found.\n"
                "2. **USE SPECIFIC NAMES**: If evidence mentions a specific sub-location (e.g., 'West Wing', 'Village X'), use that name in the query instead of the general area.\n"
                "3. **SHORT & PRECISE**: Keep query <= 8 words.\n\n"
                "Return JSON with keys: scratchpad_update, enough, missing, next_query."
            )
            ),
            (
                "user",
                "Question:\n{question}\n\n"
                "Scratchpad:\n{scratchpad}\n\n"
                "History (previous queries & docs):\n{history}\n\n"
                "Current Evidence:\n{evidence}\n",
            ),
        ]
    )

    def parse_json(s: str):
        try:
            obj = json.loads(s.strip())
            return {
                "scratchpad_update": str(obj.get("scratchpad_update", "")).strip(),
                "enough": bool(obj.get("enough", False)),
                "missing": str(obj.get("missing", "")).strip(),
                "next_query": str(obj.get("next_query", "")).strip(),
            }
        except Exception:
            return {
                "scratchpad_update": s.strip(),
                "enough": False,
                "missing": "",
                "next_query": "",
            }

    return (
        step_prompt
        | llm.bind(temperature=0)
        | StrOutputParser()
        | RunnableLambda(parse_json)
    )


final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using ONLY the provided evidence.\n"
            "Do not add background knowledge or plausible details.\n"
            "Do not infer actions/events unless explicitly stated.\n"
            "Prefer hedged phrasing for interpretations (e.g., 'is likened to').\n"
            "If evidence is insufficient for any part, say so.",
        ),
        ("user", "Question: {question}\n\nEvidence:\n{context}\n\n"),
    ]
)


def build_cot_rag_chain(
    retriever: Retriever,
    llm,
    cot_cfg: CoTRAGConfig = CoTRAGConfig(),
):
    step_reasoner = make_step_reasoner(llm)
    answer_chain = final_prompt | llm | StrOutputParser()

    def dbg(title: str, body: str = ""):
        if not cot_cfg.verbose:
            return
        line = "=" * 70
        print(f"\n{line}\n{title}\n{line}")
        if body:
            print(body)

    def fmt_docs(docs: List[Dict[str, Any]], max_items: int):
        lines = []
        for i, d in enumerate(docs[:max_items], start=1):
            preview = shorten(
                d["text"].replace("\n", " "),
                width=cot_cfg.preview_chars,
                placeholder="…",
            )
            prior = d.get("prior_rrf", None)
            prior_txt = f" | prior={prior:.4f}" if isinstance(prior, (int, float)) else ""
            ppr = (d.get("metadata") or {}).get("score_ppr", None)
            ppr_txt = f" | ppr={ppr:.4f}" if isinstance(ppr, (int, float)) else ""
            lines.append(f"{i:02d}. [{d['id']}] {preview}{prior_txt}{ppr_txt}")
        return "\n".join(lines)

    def select_step_docs(
        ranking: List[Dict[str, Any]],
        seen_ids: set,
        k: int,
        dedup: bool = True,
    ) -> List[Dict[str, Any]]:
        if not dedup:
            return ranking[:k]
        picked: List[Dict[str, Any]] = []
        for d in ranking:
            if d["id"] in seen_ids:
                continue
            picked.append(d)
            seen_ids.add(d["id"])
            if len(picked) >= k:
                break
        return picked

    def build_evidence_text_full(docs: List[Dict[str, Any]]) -> str:
        parts = []
        for d in docs:
            parts.append(f"[{d['id']}]\n{d['text']}")
        return "\n\n---\n\n".join(parts)

    def doc_score(d: Dict[str, Any]) -> float:
        meta = d.get("metadata") or {}
        ppr = meta.get("score_ppr", None)
        if isinstance(ppr, (int, float)):
            return float(ppr)
        prior = d.get("prior_rrf", None)
        if isinstance(prior, (int, float)):
            return float(prior)
        return 0.0

    def select_final_docs(all_evidence: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        for d in all_evidence:
            by_id.setdefault(d["id"], d)
        order_index = {d["id"]: i for i, d in enumerate(all_evidence)}
        candidates = list(by_id.values())
        candidates.sort(
            key=lambda d: (doc_score(d), -order_index.get(d["id"], 10**9)),
            reverse=True,
        )
        return candidates[:k]

    def _pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
        q0 = inputs["question"]
        dbg("1) QUERY ORIGINALE (q0)", q0)

        scratchpad = ""
        all_evidence: List[Dict[str, Any]] = []
        seen_ids = set()
        history: List[Dict[str, Any]] = []
        current_query = q0

        for t in range(1, cot_cfg.max_steps + 1):
            dbg(f"2.{t}) RETRIEVAL | query='{current_query}'")

            ranking = retriever.retrieve(current_query, cot_cfg.k_pool_per_step)
            if not ranking:
                break

            dbg(f"2.{t}) RANKING (top 5)", fmt_docs(ranking, 5))

            step_docs = select_step_docs(
                ranking=ranking,
                seen_ids=seen_ids,
                k=cot_cfg.top_k_step_context,
                dedup=cot_cfg.dedup_across_steps,
            )

            all_evidence.extend(step_docs)

            history.append(
                {
                    "step": t,
                    "query": current_query,
                    "doc_ids": [d["id"] for d in step_docs],
                }
            )
            hist_slice = history[-cot_cfg.history_window :]
            history_text = json.dumps(hist_slice, ensure_ascii=False)

            ev_text = build_evidence_text_full(step_docs)
            dbg(f"2.{t}) EVIDENCE (full)", ev_text[:2000] + ("\n...\n" if len(ev_text) > 2000 else ""))

            upd = step_reasoner.invoke(
                {
                    "question": q0,
                    "scratchpad": scratchpad,
                    "history": history_text,
                    "evidence": ev_text,
                }
            )

            if upd["scratchpad_update"]:
                scratchpad = (scratchpad + "\n" + upd["scratchpad_update"]).strip()

            dbg(f"2.{t}) SCRATCHPAD UPDATE", upd["scratchpad_update"])
            dbg(
                f"2.{t}) STOP?",
                f"enough={upd['enough']} | missing={upd['missing']} | next_query={upd['next_query']}",
            )

            if cot_cfg.stop_if_enough and upd["enough"]:
                break

            next_q = upd["next_query"].strip()
            current_query = next_q if next_q else q0

        final_docs = select_final_docs(all_evidence, cot_cfg.top_k_final)
        dbg("3) FINAL DOCS", fmt_docs(final_docs, cot_cfg.top_k_final))

        context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in final_docs)
        answer = answer_chain.invoke({"question": q0, "context": context})

        return {
            "question": q0,
            "answer": answer,
            "documents": final_docs,
            "context": context,
            "scratchpad": scratchpad,
            "history": history,
        }

    return RunnableLambda(_pipeline)


def build_hybrid_retriever_from_cfg(cfg: PipelineConfig) -> HybridRetriever:
    chroma_cfg = cfg.chroma
    client_chroma = chromadb.Client(
        Settings(
            is_persistent=True,
            persist_directory=chroma_cfg.persist_directory,
        )
    )
    col = client_chroma.get_collection(chroma_cfg.collection_name_chunks)
    model = SentenceTransformer(cfg.embeddings.model_name)

    return HybridRetriever(
        chroma_collection=col,
        model=model,
        rrf_k=cfg.hybrid.rrf_k,
        w_dense=cfg.hybrid.w_dense,
        w_sparse=cfg.hybrid.w_sparse,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    cfg = DEFAULT_CONFIG
    llm = get_llm_client(cfg)
    hybrid = build_hybrid_retriever_from_cfg(cfg)
    driver = GraphDatabase.driver(
        cfg.neo4j.uri,
        auth=(cfg.neo4j.user, cfg.neo4j.password),
    )
    cot_cfg = CoTRAGConfig(
        k_per_step_dense=10,
        k_per_step_sparse=10,
        k_pool_per_step=25,
        top_k_step_context=5,
        max_steps=4,
        stop_if_enough=True,
        top_k_final=6,
        verbose=True,
        history_window=2,
    )
    mode = "graph"
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()
    if mode == "classic":
        backend: Retriever = ClassicHybridBackend(hybrid, cot_cfg)
    else:
        backend = GraphRAGBackend(driver, hybrid, cfg, cot_cfg)
    chain = build_cot_rag_chain(backend, llm, cot_cfg)
    question = "Which events connect the Portuguese Embassador’s leave-taking to my Lord Embassador’s visit to Trinity House?"
    result = chain.invoke({"question": question})
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== CONTEXT (doc ids) ===")
    for d in result["documents"]:
        meta = d.get("metadata", {})
        print(
            f"- {d['id']} | doc={meta.get('doc_id')} pos={meta.get('position')} comm={meta.get('community_id')}"
        )
