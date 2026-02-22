from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from elasticsearch import Elasticsearch
from neo4j import GraphDatabase, Session

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEFAULT_CONFIG, PipelineConfig  # noqa: E402

logger = logging.getLogger(__name__)


def build_es_client_from_cfg(cfg: PipelineConfig) -> Elasticsearch:
    user = getattr(cfg.es, "user", None)
    pwd = getattr(cfg.es, "password", None)
    basic_auth = (user, pwd) if (user and pwd) else None
    return Elasticsearch(
        cfg.es.url,
        basic_auth=basic_auth,
        verify_certs=getattr(cfg.es, "verify_certs", True),
    )


def create_next_edges(session: Session) -> None:
    query = """
    MATCH (c:Chunk)
    WITH c ORDER BY c.doc_id, c.position
    WITH c.doc_id AS doc_id, collect(c) AS chunks
    UNWIND range(0, size(chunks) - 2) AS i
    WITH chunks[i] AS a, chunks[i + 1] AS b
    MERGE (a)-[:NEXT]->(b)
    """
    session.run(query)


def create_next_edges_for_doc(session: Session, doc_id: str) -> None:
    query = """
    MATCH (c:Chunk {doc_id: $doc_id})
    WITH c ORDER BY c.position
    WITH collect(c) AS chunks
    UNWIND range(0, size(chunks) - 2) AS i
    WITH chunks[i] AS a, chunks[i + 1] AS b
    MERGE (a)-[:NEXT]->(b)
    """
    session.run(query, doc_id=doc_id)


def iter_es_doc_vectors(
    es: Elasticsearch,
    index_name: str,
    doc_id: str,
    page_size: int = 2000,
) -> List[Tuple[str, List[float]]]:
    out: List[Tuple[str, List[float]]] = []
    sort = [{"chunk_id": "asc"}]
    search_after = None

    while True:
        body: Dict[str, Any] = {
            "size": page_size,
            "query": {"term": {"doc_id": doc_id}},
            "_source": ["chunk_id", "vector"],
            "sort": sort,
        }
        if search_after is not None:
            body["search_after"] = search_after

        resp = es.search(index=index_name, body=body)
        hits = resp.get("hits", {}).get("hits", []) or []
        if not hits:
            break

        for h in hits:
            src = h.get("_source", {}) or {}
            cid = src.get("chunk_id") or h.get("_id")
            vec = src.get("vector")
            if cid and isinstance(vec, list) and vec:
                out.append((cid, vec))

        search_after = hits[-1].get("sort")
        if not search_after:
            break

    return out


def mget_vectors_by_chunk_ids(
    es: Elasticsearch,
    index_name: str,
    chunk_ids: List[str],
) -> Dict[str, List[float]]:
    if not chunk_ids:
        return {}

    resp = es.mget(
        index=index_name,
        body={"ids": chunk_ids, "_source": ["chunk_id", "vector"]},
    )
    out: Dict[str, List[float]] = {}
    for d in resp.get("docs", []) or []:
        if not d.get("found"):
            continue
        src = d.get("_source", {}) or {}
        cid = src.get("chunk_id") or d.get("_id")
        vec = src.get("vector")
        if cid and isinstance(vec, list) and vec:
            out[cid] = vec
    return out


def knn_neighbor_ids_for_chunk(
    es: Elasticsearch,
    index_name: str,
    doc_id: str,
    chunk_id: str,
    vector: List[float],
    k: int,
    num_candidates: int,
) -> List[str]:
    body_knn = {
        "size": k + 1,
        "query": {"bool": {"filter": [{"term": {"doc_id": doc_id}}]}},
        "knn": {
            "field": "vector",
            "query_vector": vector,
            "k": k + 1,
            "num_candidates": max(num_candidates, (k + 1) * 10),
        },
        "_source": ["chunk_id"],
    }

    try:
        resp = es.search(index=index_name, body=body_knn)
    except Exception:
        body_script = {
            "size": k + 1,
            "query": {
                "script_score": {
                    "query": {"bool": {"filter": [{"term": {"doc_id": doc_id}}]}},
                    "script": {
                        "source": "cosineSimilarity(params.qvec, 'vector') + 1.0",
                        "params": {"qvec": vector},
                    },
                }
            },
            "_source": ["chunk_id"],
        }
        resp = es.search(index=index_name, body=body_script)

    hits = resp.get("hits", {}).get("hits", []) or []
    nbrs: List[str] = []
    for h in hits:
        src = h.get("_source", {}) or {}
        cid = src.get("chunk_id") or h.get("_id")
        if not cid or cid == chunk_id:
            continue
        nbrs.append(cid)
        if len(nbrs) >= k:
            break
    return nbrs


def _write_similarity_edges_batch(session: Session, edges: List[Dict[str, Any]]) -> None:
    if not edges:
        return
    query = """
    UNWIND $edges AS edge
    MATCH (a:Chunk {chunk_id: edge.src})
    MATCH (b:Chunk {chunk_id: edge.tgt})
    MERGE (a)-[r:SIMILAR_TO]->(b)
      ON CREATE SET r.sim_score = edge.score
      ON MATCH  SET r.sim_score = edge.score
    """
    session.run(query, edges=edges)


def create_similarity_edges_es_for_doc(
    session: Session,
    es: Elasticsearch,
    config: PipelineConfig,
    doc_id: str,
) -> int:
    chunks_index = config.es.index_name
    top_k = int(config.similarity.top_k_neighbors)
    thr = float(config.similarity.similarity_threshold)

    matrix_max = int(getattr(config.similarity, "matrix_max_chunks", 2000))
    es_page = int(getattr(config.similarity, "es_page_size", 2000))
    num_candidates = int(getattr(config.similarity, "num_candidates", max(top_k * 10, 100)))
    edge_batch = int(getattr(config.similarity, "neo4j_edge_batch", 5000))

    pairs = iter_es_doc_vectors(es, chunks_index, doc_id=doc_id, page_size=es_page)
    if len(pairs) < 2:
        logger.info("doc_id=%s -> not enough chunks (%d) for SIMILAR_TO", doc_id, len(pairs))
        return 0

    ids = [p[0] for p in pairs]
    vectors = [p[1] for p in pairs]

    local_edges: List[Dict[str, Any]] = []

    if len(ids) <= matrix_max:
        mat = np.array(vectors, dtype=np.float32)
        sim = cosine_similarity(mat)
        n = len(ids)
        for i in range(n):
            neighbors = [(j, float(sim[i][j])) for j in range(n) if j != i]
            neighbors.sort(key=lambda x: x[1], reverse=True)
            for j, score in neighbors[:top_k]:
                if score >= thr:
                    local_edges.append({"src": ids[i], "tgt": ids[j], "score": score})
    else:
        vec_map = {cid: np.array(vec, dtype=np.float32) for cid, vec in pairs}
        for cid, vec in pairs:
            nbr_ids = knn_neighbor_ids_for_chunk(
                es=es,
                index_name=chunks_index,
                doc_id=doc_id,
                chunk_id=cid,
                vector=vec,
                k=top_k,
                num_candidates=num_candidates,
            )
            nbr_vecs = mget_vectors_by_chunk_ids(es, chunks_index, nbr_ids)
            a = vec_map[cid]
            a_norm = np.linalg.norm(a) + 1e-12
            for nid in nbr_ids:
                bv = nbr_vecs.get(nid)
                if bv is None:
                    continue
                b = np.array(bv, dtype=np.float32)
                score = float(np.dot(a, b) / (a_norm * (np.linalg.norm(b) + 1e-12)))
                if score >= thr:
                    local_edges.append({"src": cid, "tgt": nid, "score": score})

    if not local_edges:
        logger.info("doc_id=%s -> SIMILAR_TO edges: 0 (threshold too high?)", doc_id)
        return 0

    for start in range(0, len(local_edges), edge_batch):
        _write_similarity_edges_batch(session, local_edges[start:start + edge_batch])

    logger.info("doc_id=%s -> SIMILAR_TO edges: %d", doc_id, len(local_edges))
    return len(local_edges)


def create_similarity_edges_es(
    session: Session,
    es: Elasticsearch,
    config: PipelineConfig = DEFAULT_CONFIG,
    doc_id: Optional[str] = None,
) -> None:
    if doc_id:
        create_similarity_edges_es_for_doc(session, es, config, doc_id)
        return

    result = session.run("MATCH (c:Chunk) RETURN DISTINCT c.doc_id AS doc_id")
    doc_ids = [record["doc_id"] for record in result]

    total = 0
    for did in doc_ids:
        total += create_similarity_edges_es_for_doc(session, es, config, did)

    logger.info("Total SIMILAR_TO edges created/updated: %d", total)


def build_graph(config: PipelineConfig = DEFAULT_CONFIG, doc_id: Optional[str] = None) -> None:
    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password),
    )
    es = build_es_client_from_cfg(config)
    if not es.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {config.es.url}")

    try:
        with driver.session(database=config.neo4j.database) as session:
            if doc_id:
                create_next_edges_for_doc(session, doc_id)
                create_similarity_edges_es(session, es, config, doc_id=doc_id)
            else:
                create_next_edges(session)
                create_similarity_edges_es(session, es, config, doc_id=None)
    finally:
        try:
            es.close()
        except Exception:
            pass
        driver.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--doc_id", type=str, default=None, help="Process only this doc_id (incremental mode).")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    build_graph(DEFAULT_CONFIG, doc_id=args.doc_id)
