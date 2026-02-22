from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from neo4j import GraphDatabase, Session

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from sentence_transformers import SentenceTransformer

from config import DEFAULT_CONFIG, PipelineConfig
from llm_stub import get_llm_client

logger = logging.getLogger(__name__)

@dataclass
class CoTRAGConfig:
    k_pool_per_step: int = 30
    top_k_step_context: int = 5
    max_steps: int = 4
    stop_if_enough: bool = True
    dedup_across_steps: bool = True
    history_window: int = 2
    top_k_final: int = 6
    verbose: bool = True
    preview_chars: int = 180
    min_steps: int = 2
    require_next_query_if_not_enough: bool = True
    history_query_sim_threshold: float = 0.85

class ElasticsearchRetriever:
    def __init__(self, client: Elasticsearch, model: SentenceTransformer, index_name: str):
        self.client = client
        self.model = model
        self.index_name = index_name

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
        search_body = {
            "size": k,
            "query": {"match": {"text": query}},
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": k * 10,
            },
            "_source": ["chunk_id", "doc_id", "text", "position", "community_id"],
        }
        response = self.client.search(index=self.index_name, body=search_body)

        results: List[Dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            src = hit.get("_source", {}) or {}
            score = float(hit.get("_score", 0.0) or 0.0)
            results.append(
                {
                    "id": src.get("chunk_id", hit.get("_id")),
                    "text": src.get("text", ""),
                    "prior_rrf": score,
                    "metadata": {
                        "doc_id": src.get("doc_id"),
                        "position": src.get("position"),
                        "community_id": src.get("community_id"),
                        "es_score": score,
                    },
                }
            )
        return results

def community_index_name(cfg: PipelineConfig) -> str:
    explicit = getattr(cfg.es, "community_index_name", None)
    return explicit or f"{cfg.es.index_name}_communities"

def get_top_communities_by_summary_es(
    es: Elasticsearch,
    model: SentenceTransformer,
    query: str,
    cfg: PipelineConfig,
    k: int,
) -> List[Dict[str, Any]]:
    index_name = community_index_name(cfg)
    qvec = model.encode(query, normalize_embeddings=True).tolist()

    body = {
        "size": k,
        "query": {"match": {"summary": query}},
        "knn": {
            "field": "vector",
            "query_vector": qvec,
            "k": k,
            "num_candidates": max(k * 10, 100),
        },
        "_source": ["community_key", "doc_id", "community_id", "summary"],
    }

    resp = es.search(index=index_name, body=body)
    hits = resp.get("hits", {}).get("hits", []) or []

    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source", {}) or {}
        ck = src.get("community_key")
        if not ck:
            continue
        out.append(
            {
                "community_key": str(ck),
                "score": float(h.get("_score", 0.0) or 0.0),
                "doc_id": src.get("doc_id"),
                "community_id": src.get("community_id"),
                "summary": src.get("summary"),
            }
        )

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def expand_communities_by_key(
    session: Session,
    seed_keys: List[str],
    hops: int = 1,
    limit: int = 30,
    min_sim: float = 0.0,
) -> List[Dict[str, Any]]:
    if not seed_keys or limit <= 0:
        return []

    hops = max(1, min(int(hops), 3))
    limit = int(limit)

    query = f"""
    MATCH (s:Community)
    WHERE s.community_key IN $seed_keys
    MATCH p=(s)-[r:COMM_SIMILAR_TO*1..{hops}]->(n:Community)
    WHERE ALL(x IN r WHERE x.sim_score >= $min_sim)
    WITH n,
         reduce(sc=1.0, x IN r | sc * x.sim_score) AS path_score
    RETURN n.community_key AS community_key, max(path_score) AS score
    ORDER BY score DESC
    LIMIT $limit
    """
    result = session.run(query, seed_keys=seed_keys, limit=limit, min_sim=float(min_sim))

    out: List[Dict[str, Any]] = []
    for rec in result:
        ck = rec.get("community_key")
        if ck:
            out.append({"community_key": str(ck), "score": float(rec["score"])})
    return out

def fetch_chunks_for_community_keys(session: Session, community_keys: List[str]) -> List[str]:
    if not community_keys:
        return []
    query = """
    MATCH (comm:Community)-[:HAS_CHUNK]->(c:Chunk)
    WHERE comm.community_key IN $cks
    RETURN DISTINCT c.chunk_id AS chunk_id
    """
    result = session.run(query, cks=community_keys)
    return [r["chunk_id"] for r in result]

def es_hybrid_retrieve_chunks_in_subset(
    es: Elasticsearch,
    model: SentenceTransformer,
    index_name: str,
    query: str,
    allowed_chunk_ids: List[str],
    k: int,
    num_candidates: int = 200,
) -> List[Dict[str, Any]]:
    if not allowed_chunk_ids or k <= 0:
        return []

    qvec = model.encode(query, normalize_embeddings=True).tolist()
    id_filter = {"terms": {"chunk_id": allowed_chunk_ids}}

    body_knn = {
        "size": k,
        "query": {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": [id_filter],
            }
        },
        "knn": {
            "field": "vector",
            "query_vector": qvec,
            "k": k,
            "num_candidates": max(num_candidates, k * 10),
        },
        "_source": ["chunk_id", "doc_id", "position", "text", "community_id"],
    }

    try:
        resp = es.search(index=index_name, body=body_knn)
    except Exception as e:
        logger.warning("ES knn+query failed (%s). Falling back to script_score.", e)
        body_script = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [{"match": {"text": query}}],
                            "filter": [id_filter],
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.qvec, 'vector') + 1.0",
                        "params": {"qvec": qvec},
                    },
                }
            },
            "_source": ["chunk_id", "doc_id", "position", "text", "community_id"],
        }
        resp = es.search(index=index_name, body=body_script)

    hits = resp.get("hits", {}).get("hits", []) or []

    docs: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source", {}) or {}
        cid = src.get("chunk_id") or h.get("_id")
        score = float(h.get("_score", 0.0) or 0.0)
        docs.append(
            {
                "id": cid,
                "text": src.get("text", "") or "",
                "prior_rrf": score,
                "metadata": {
                    "doc_id": src.get("doc_id"),
                    "position": src.get("position"),
                    "community_id": src.get("community_id"),
                    "es_score": score,
                    "score_graph": score,
                },
            }
        )
    return docs

def graph_rag_retrieve_v2(
    session: Session,
    retriever: ElasticsearchRetriever,
    query: str,
    cfg: PipelineConfig,
    max_results: int,
) -> List[Dict[str, Any]]:
    seed_k = int(getattr(cfg, "query_top_k_communities", 10))
    hops = int(getattr(getattr(cfg, "community_expansion", object()), "hops", 1))
    expand_limit = int(getattr(getattr(cfg, "community_expansion", object()), "max_expanded_communities", 30))
    expand_min_sim = float(getattr(getattr(cfg, "community_expansion", object()), "min_edge_sim", 0.0))

    comms = get_top_communities_by_summary_es(
        es=retriever.client,
        model=retriever.model,
        query=query,
        cfg=cfg,
        k=seed_k,
    )
    if not comms:
        return []

    seed_keys = [c["community_key"] for c in comms[:seed_k] if c.get("community_key")]
    if not seed_keys:
        return []

    expanded = expand_communities_by_key(
        session=session,
        seed_keys=seed_keys,
        hops=hops,
        limit=expand_limit,
        min_sim=expand_min_sim,
    )
    expanded_keys = [x["community_key"] for x in expanded if x.get("community_key")]
    all_keys = list(dict.fromkeys(seed_keys + expanded_keys))

    allowed_chunk_ids = fetch_chunks_for_community_keys(session, all_keys)
    if not allowed_chunk_ids:
        return []

    docs = es_hybrid_retrieve_chunks_in_subset(
        es=retriever.client,
        model=retriever.model,
        index_name=cfg.es.index_name,
        query=query,
        allowed_chunk_ids=allowed_chunk_ids,
        k=max_results,
        num_candidates=max(max_results * 10, 100),
    )

    for d in docs:
        dmeta = d.setdefault("metadata", {})
        dmeta["seed_community_keys"] = seed_keys[: min(len(seed_keys), 10)]
        dmeta["expanded_communities_count"] = max(0, len(all_keys) - len(seed_keys))
        dmeta["allowed_chunk_pool_size"] = len(allowed_chunk_ids)

    return [d for d in docs if d.get("text")]

class ClassicHybridBackend:
    def __init__(self, retriever: ElasticsearchRetriever, cfg_cot: CoTRAGConfig):
        self.retriever = retriever
        self.cfg_cot = cfg_cot

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        return self.retriever.retrieve(query, k)

class GraphRAGBackend:
    def __init__(self, driver, retriever: ElasticsearchRetriever, graph_cfg: PipelineConfig, cot_cfg: CoTRAGConfig):
        self.driver = driver
        self.retriever = retriever
        self.graph_cfg = graph_cfg
        self.cot_cfg = cot_cfg

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.graph_cfg.neo4j.database) as session:
            return graph_rag_retrieve_v2(
                session=session,
                retriever=self.retriever,
                query=query,
                cfg=self.graph_cfg,
                max_results=k,
            )

def _norm_tokens(q: str) -> List[str]:
    q = q.lower()
    q = re.sub(r"[^\w\s]", " ", q)
    toks = [t for t in q.split() if t]
    return toks[:32]

def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 0.0

def _is_too_similar(new_q: str, prev_qs: List[str], thr: float) -> bool:
    nt = _norm_tokens(new_q)
    for pq in prev_qs:
        if _jaccard(nt, _norm_tokens(pq)) >= thr:
            return True
    return False

def make_step_reasoner(llm):
    step_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strategic retrieval controller for multi-hop reasoning.\n"
                "Output ONLY valid JSON. Do not use tools.\n\n"
                "You must update the scratchpad as SHORT BULLET FACTS grounded in the evidence.\n"
                "Scratchpad format rules (strict):\n"
                "- scratchpad_update MUST be a single STRING.\n"
                "- Use 1-6 lines, each starting with '- '.\n"
                "- Each line must end with a source id in parentheses, e.g. (Novel-xxxx_c_yy).\n"
                "- Do NOT output JSON, dicts, YAML, code blocks, or nested structures inside scratchpad_update.\n"
                "- Do NOT ask for information not present in the corpus unless evidence explicitly indicates such a value exists.\n\n"
                "Entity bridging method:\n"
                "1) Decompose the question into atomic sub-questions.\n"
                "2) Extract salient entities from evidence.\n"
                "3) If question links A->C and evidence supports A->B, treat B as a Bridge Entity and search B->C next.\n\n"
                "Coverage / stop rules:\n"
                "- enough=true ONLY if the current evidence supports all sub-questions at the level required by the question.\n"
                "- If the evidence provides a qualitative relation that answers the question (e.g., 'near', 'in', 'by', 'toward'), that is sufficient unless the question explicitly requests a numeric/exact measure.\n"
                "- If not enough, set enough=false and list missing as short phrases.\n"
                "- When enough=false you MUST provide next_query (<= 8 words) that targets ONLY the missing part.\n"
                "- next_query must include at least one specific entity from the evidence.\n"
                "- next_query must NOT be a trivial reordering of a previous query in History.\n\n"
                "Return ONLY valid JSON with keys:\n"
                "scratchpad_update (string), enough (boolean), missing (string or list), next_query (string).",
            ),
            (
                "user",
                "Question:\n{question}\n\nScratchpad:\n{scratchpad}\n\nHistory:\n{history}\n\nCurrent Evidence:\n{evidence}\n",
            ),
        ]
    )

    def extract_json(text: str) -> str:
        t = text.strip()
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        return m.group(0) if m else t

    def to_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            x = v.strip().lower()
            if x in {"true", "yes", "1"}:
                return True
            if x in {"false", "no", "0", ""}:
                return False
        return False

    def to_missing_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            s = v.strip()
            if not s or s == "[]":
                return []
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                if not inner:
                    return []
                parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
                return [p for p in parts if p]
            return [s]
        return [str(v).strip()] if str(v).strip() else []

    def normalize_scratchpad_update(x: Any) -> str:
        s = str(x or "").strip()
        s = re.sub(r"\r\n?", "\n", s)
        lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
        out = []
        for ln in lines:
            if ln.startswith("{") or ln.startswith("[") or ln.endswith("}") or ln.endswith("]"):
                continue
            if ln.lower().startswith(("scratchpad_update", "enough", "missing", "next_query")):
                continue
            if not ln.startswith("- "):
                ln = "- " + ln
            out.append(ln)
            if len(out) >= 6:
                break
        return "\n".join(out).strip()

    def parse_json(s: str):
        raw = extract_json(s)
        try:
            obj = json.loads(raw)
            missing_list = to_missing_list(obj.get("missing", ""))
            enough = to_bool(obj.get("enough", False))
            if missing_list:
                enough = False
            sp = normalize_scratchpad_update(obj.get("scratchpad_update", ""))
            return {
                "scratchpad_update": sp,
                "enough": enough,
                "missing": missing_list,
                "next_query": str(obj.get("next_query", "")).strip(),
            }
        except Exception:
            sp = normalize_scratchpad_update(s)
            return {"scratchpad_update": sp, "enough": False, "missing": [], "next_query": ""}

    return step_prompt | llm.bind(temperature=0) | StrOutputParser() | RunnableLambda(parse_json)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using ONLY the provided evidence.\nIf evidence is insufficient for any part, say so."),
        ("user", "Question: {question}\n\nEvidence:\n{context}\n\n"),
    ]
)

def build_cot_rag_chain(retriever, llm, cot_cfg: CoTRAGConfig = CoTRAGConfig()):
    step_reasoner = make_step_reasoner(llm)
    answer_chain = final_prompt | llm | StrOutputParser()

    def dbg(title: str, body: str = ""):
        if not cot_cfg.verbose:
            return
        print(f"\n{'='*70}\n{title}\n{'='*70}")
        if body:
            print(body)

    def build_evidence_text_full(docs: List[Dict[str, Any]]) -> str:
        return "\n\n---\n\n".join([f"[{d['id']}]\n{d['text']}" for d in docs])

    def dedup_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for d in docs:
            cid = d.get("id")
            if cid in seen:
                continue
            seen.add(cid)
            out.append(d)
        return out

    def select_final_docs(all_evidence: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        by_id = {}
        for d in all_evidence:
            by_id[d["id"]] = d
        candidates = list(by_id.values())
        candidates.sort(
            key=lambda d: d.get("metadata", {}).get(
                "score_graph",
                d.get("metadata", {}).get("score_ppr", d.get("prior_rrf", 0.0)),
            ),
            reverse=True,
        )
        return candidates[:k]

    def _pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
        q0 = inputs["question"]
        scratchpad = ""
        all_evidence: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []
        current_query = q0

        for t in range(1, cot_cfg.max_steps + 1):
            dbg(f"STEP {t}: Retrieving", current_query)
            ranking = retriever.retrieve(current_query, cot_cfg.k_pool_per_step)
            if not ranking:
                break

            step_docs = ranking[: cot_cfg.top_k_step_context]
            all_evidence.extend(step_docs)

            if cot_cfg.dedup_across_steps:
                all_evidence = dedup_docs(all_evidence)

            history.append({"step": t, "query": current_query, "doc_ids": [d["id"] for d in step_docs]})
            ev_text = build_evidence_text_full(step_docs)
            hist_json = json.dumps(history[-cot_cfg.history_window :], ensure_ascii=False)

            upd = step_reasoner.invoke(
                {"question": q0, "scratchpad": scratchpad, "history": hist_json, "evidence": ev_text}
            )

            if upd["scratchpad_update"]:
                scratchpad = (scratchpad + "\n" + upd["scratchpad_update"]).strip()

            dbg(
                f"STEP {t}: Reasoning",
                f"Next: {upd['next_query']} | Enough: {upd['enough']} | Missing: {upd['missing']}",
            )

            if t < cot_cfg.min_steps:
                if upd["next_query"]:
                    prev_qs = [h["query"] for h in history[-cot_cfg.history_window :]]
                    if _is_too_similar(upd["next_query"], prev_qs, cot_cfg.history_query_sim_threshold):
                        break
                    current_query = upd["next_query"]
                    continue
                break

            if cot_cfg.stop_if_enough and upd["enough"]:
                break

            if not upd["enough"]:
                next_q = upd["next_query"]
                if cot_cfg.require_next_query_if_not_enough and not next_q:
                    break
                if next_q:
                    prev_qs = [h["query"] for h in history[-cot_cfg.history_window :]]
                    if _is_too_similar(next_q, prev_qs, cot_cfg.history_query_sim_threshold):
                        break
                    current_query = next_q
                    continue
                break
            break

        final_docs = select_final_docs(all_evidence, cot_cfg.top_k_final)
        context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in final_docs)
        answer = answer_chain.invoke({"question": q0, "context": context})

        return {
            "question": q0,
            "answer": answer,
            "documents": final_docs,
            "scratchpad": scratchpad,
            "history": history,
        }

    return RunnableLambda(_pipeline)

def build_es_retriever_from_cfg(cfg: PipelineConfig) -> ElasticsearchRetriever:
    es_client = Elasticsearch(
        cfg.es.url,
        basic_auth=(cfg.es.user, cfg.es.password) if getattr(cfg.es, "user", None) else None,
        verify_certs=getattr(cfg.es, "verify_certs", True),
    )
    model = SentenceTransformer(cfg.embeddings.model_name)
    return ElasticsearchRetriever(client=es_client, model=model, index_name=cfg.es.index_name)

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    cfg = DEFAULT_CONFIG
    llm = get_llm_client(cfg)
    es_retriever = build_es_retriever_from_cfg(cfg)
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))

    cot_cfg = CoTRAGConfig(max_steps=3, min_steps=2, verbose=True)

    mode = "classic"
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()

    if mode == "classic":
        backend = ClassicHybridBackend(es_retriever, cot_cfg)
    else:
        backend = GraphRAGBackend(driver, es_retriever, cfg, cot_cfg)

    chain = build_cot_rag_chain(backend, llm, cot_cfg)

    question = (
        "How does the narrator in 'An Unsentimental Journey through Cornwall' compare Mont St. Michel in Normandy "
        "to St. Michael's Mount in Cornwall, and what similarities are noted between the two locations?"
    )
    result = chain.invoke({"question": question})

    print("\n=== ANSWER ===")
    print(result["answer"])