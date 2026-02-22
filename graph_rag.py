from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional

from neo4j import GraphDatabase, Session
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from elasticsearch import Elasticsearch

from config import DEFAULT_CONFIG, PipelineConfig
from llm_stub import get_llm_client

logger = logging.getLogger(__name__)
cfg = DEFAULT_CONFIG
llm = get_llm_client(cfg)

@lru_cache(maxsize=None)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

@lru_cache(maxsize=None)
def get_es_client(
    url: str,
    user: Optional[str],
    password: Optional[str],
    verify_certs: bool,
) -> Elasticsearch:
    basic_auth = (user, password) if (user and password) else None
    return Elasticsearch(
        url,
        basic_auth=basic_auth,
        verify_certs=verify_certs,
    )

def community_index_name(config: PipelineConfig) -> str:
    return f"{config.es.index_name}_communities"

answer_prompt = ChatPromptTemplate.from_messages(
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
            "user",
            (
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
        ),
    ]
)

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
    result = session.run(
        query,
        seed_keys=seed_keys,
        limit=limit,
        min_sim=float(min_sim),
    )

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

def fetch_chunk_metadata(session: Session, chunk_ids: List[str]) -> Dict[str, dict]:
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

def fetch_community_summaries_by_key(session: Session, community_keys: List[str]) -> Dict[str, str]:
    if not community_keys:
        return {}

    query = """
    MATCH (comm:Community)
    WHERE comm.community_key IN $cks
    RETURN comm.community_key AS ck, comm.summary AS summary
    """
    result = session.run(query, cks=community_keys)
    out: Dict[str, str] = {}
    for r in result:
        ck = r.get("ck")
        summ = r.get("summary")
        if ck and summ:
            out[str(ck)] = str(summ)
    return out

def build_context(top_chunks: List[Tuple[str, float]], metadata: Dict[str, dict]) -> str:
    evidence_lines: List[str] = []
    for chunk_id, score in top_chunks:
        meta = metadata.get(chunk_id)
        if not meta:
            continue
        evidence_lines.append(
            f"- [{meta['doc_id']}#{meta['position']}] (score={score:.4f}) {meta['text']}"
        )

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "No evidence passages available."
    return "Evidence passages:\n" + evidence_block

def es_hybrid_retrieve_chunks_in_subset(
    es: Elasticsearch,
    model: SentenceTransformer,
    index_name: str,
    query: str,
    allowed_chunk_ids: List[str],
    k: int,
    num_candidates: int = 200,
) -> List[Dict[str, Any]]:
    if not allowed_chunk_ids:
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
        "_source": ["chunk_id", "doc_id", "position", "text"],
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
            "_source": ["chunk_id", "doc_id", "position", "text"],
        }
        resp = es.search(index=index_name, body=body_script)

    hits = resp.get("hits", {}).get("hits", []) or []
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source", {}) or {}
        cid = src.get("chunk_id") or h.get("_id")
        out.append({"id": cid, "score": float(h.get("_score", 0.0))})
    return out

def get_top_communities_by_summary_es(
    es: Elasticsearch,
    model: SentenceTransformer,
    query: str,
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    index_name = community_index_name(config)
    k = int(getattr(config, "query_top_k_communities", 10))
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
        "_source": ["community_key", "doc_id", "community_id"],
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
                "id": h.get("_id"),
                "score": float(h.get("_score", 0.0)) if h.get("_score") is not None else 0.0,
                "community_key": str(ck),
                "doc_id": src.get("doc_id"),
                "community_id": src.get("community_id"),
            }
        )

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def answer_query(
    query: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    es_client: Optional[Elasticsearch] = None,
    neo4j_driver=None,
) -> Dict[str, Any]:
    model = get_embedding_model(config.embeddings.model_name)

    own_es = es_client is None
    own_driver = neo4j_driver is None

    es = es_client or get_es_client(
        url=config.es.url,
        user=getattr(config.es, "user", None),
        password=getattr(config.es, "password", None),
        verify_certs=getattr(config.es, "verify_certs", True),
    )

    driver = neo4j_driver or GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password),
    )

    seed_k = int(getattr(config, "query_top_k_communities", 10))
    hops = int(getattr(getattr(config, "community_expansion", object()), "hops", 1))
    expand_limit = int(getattr(getattr(config, "community_expansion", object()), "max_expanded_communities", 30))
    expand_min_sim = float(getattr(getattr(config, "community_expansion", object()), "min_edge_sim", 0.0))
    k_chunks = int(getattr(config, "query_top_k_chunks", 8))

    try:
        community_results = get_top_communities_by_summary_es(
            es=es,
            model=model,
            query=query,
            config=config,
        )
        if not community_results:
            return {"question": query, "answer": "No info found.", "context": "", "chunks": [], "communities": []}

        top_communities = community_results[:seed_k]
        seed_keys = [c["community_key"] for c in top_communities if c.get("community_key")]

        if not seed_keys:
            return {"question": query, "answer": "No info found.", "context": "", "chunks": [], "communities": []}

        with driver.session(database=config.neo4j.database) as session:
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
                return {"question": query, "answer": "No info found.", "context": "", "chunks": [], "communities": []}

            chunk_results = es_hybrid_retrieve_chunks_in_subset(
                es=es,
                model=model,
                index_name=config.es.index_name,
                query=query,
                allowed_chunk_ids=allowed_chunk_ids,
                k=k_chunks,
                num_candidates=max(k_chunks * 10, 100),
            )
            if not chunk_results:
                return {"question": query, "answer": "No info found.", "context": "", "chunks": [], "communities": []}

            top_ids = [r["id"] for r in chunk_results]
            meta = fetch_chunk_metadata(session, top_ids)
            community_summaries = fetch_community_summaries_by_key(session, all_keys)

        top_chunks_scored = [(r["id"], r["score"]) for r in chunk_results]
        context_prompt = build_context(top_chunks=top_chunks_scored, metadata=meta)

        messages = answer_prompt.format_messages(context=context_prompt, question=query)
        response = llm.invoke(messages)
        answer_text = getattr(response, "content", str(response))

        chunks_json = []
        for r in chunk_results:
            cid = r["id"]
            m = meta.get(cid, {})
            chunks_json.append(
                {
                    "chunk_id": cid,
                    "score": r["score"],
                    "doc_id": m.get("doc_id"),
                    "text": m.get("text"),
                    "community_id": m.get("community_id"),
                }
            )

        seed_set = set(seed_keys)
        expanded_score_map = {x["community_key"]: float(x["score"]) for x in expanded if x.get("community_key")}

        communities_json: List[Dict[str, Any]] = []
        for ck in all_keys:
            summ = community_summaries.get(ck)
            if not summ:
                continue
            communities_json.append(
                {
                    "community_key": ck,
                    "summary": summ,
                    "is_seed": ck in seed_set,
                    "expand_score": expanded_score_map.get(ck, None),
                }
            )

        return {
            "question": query,
            "answer": answer_text,
            "chunks": chunks_json,
            "communities": communities_json,
            "context": context_prompt,
            "debug": {
                "seed_community_keys": seed_keys,
                "expanded_community_keys": expanded_keys,
                "all_community_keys": all_keys,
                "allowed_chunk_pool_size": len(allowed_chunk_ids),
                "es_index": config.es.index_name,
                "es_community_index": community_index_name(config),
                "community_hops": hops,
                "expand_limit": expand_limit,
                "expand_min_sim": expand_min_sim,
            },
        }

    finally:
        if own_driver:
            driver.close()
        if own_es:
            try:
                es.close()
            except Exception:
                pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q = "What might explain the shared interest of Senator Plumb and Robert Ingersoll in the Ivanhoe mine?"
    result = answer_query(q, cfg)
    print("Answer:\n", result["answer"])
    print("Debug:", result.get("debug"))