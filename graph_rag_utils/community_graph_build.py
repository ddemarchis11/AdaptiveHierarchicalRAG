from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
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


def community_index_name(cfg: PipelineConfig) -> str:
    explicit = getattr(cfg.es, "community_index_name", None)
    return explicit or f"{cfg.es.index_name}_communities"


def ensure_neo4j_constraints(session: Session) -> None:
    session.run(
        """
        CREATE CONSTRAINT community_key_unique IF NOT EXISTS
        FOR (c:Community) REQUIRE c.community_key IS UNIQUE
        """
    )
    session.run(
        """
        CREATE INDEX community_doc_id_idx IF NOT EXISTS
        FOR (c:Community) ON (c.doc_id)
        """
    )


def fetch_all_communities(session: Session) -> List[Dict[str, Any]]:
    rows = session.run(
        """
        MATCH (c:Community)
        RETURN c.community_key AS community_key, c.doc_id AS doc_id
        """
    )
    out = []
    for r in rows:
        ck = r.get("community_key")
        did = r.get("doc_id")
        if ck and did is not None:
            out.append({"community_key": ck, "doc_id": did})
    return out


def fetch_communities_for_doc(session: Session, doc_id: str) -> List[Dict[str, Any]]:
    rows = session.run(
        """
        MATCH (c:Community {doc_id: $doc_id})
        RETURN c.community_key AS community_key, c.doc_id AS doc_id
        """,
        doc_id=doc_id,
    )
    out = []
    for r in rows:
        ck = r.get("community_key")
        did = r.get("doc_id")
        if ck and did is not None:
            out.append({"community_key": ck, "doc_id": did})
    return out


def es_get_community_vector(
    es: Elasticsearch,
    comm_index: str,
    community_key: str,
) -> Optional[List[float]]:
    es_id = f"community_{community_key}"

    try:
        doc = es.get(index=comm_index, id=es_id)
        src = doc.get("_source", {}) or {}
        vec = src.get("vector")
        if isinstance(vec, list) and vec:
            return vec
    except Exception:
        pass

    try:
        resp = es.search(
            index=comm_index,
            body={
                "size": 1,
                "query": {"term": {"community_key": community_key}},
                "_source": ["vector"],
            },
        )
        hits = resp.get("hits", {}).get("hits", []) or []
        if hits:
            vec = (hits[0].get("_source", {}) or {}).get("vector")
            if isinstance(vec, list) and vec:
                return vec
    except Exception:
        pass

    return None


def es_knn_candidates(
    es: Elasticsearch,
    comm_index: str,
    query_vector: List[float],
    k: int,
    num_candidates: int,
) -> List[Dict[str, Any]]:
    body = {
        "size": k,
        "knn": {
            "field": "vector",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": max(num_candidates, k * 10),
        },
        "_source": ["community_key", "doc_id", "vector"],
    }
    resp = es.search(index=comm_index, body=body)
    return resp.get("hits", {}).get("hits", []) or []


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)


def reset_comm_sim_edges(session: Session) -> None:
    session.run("MATCH (:Community)-[r:COMM_SIMILAR_TO]->(:Community) DELETE r")


def write_comm_edges(session: Session, edges: List[Dict[str, Any]]) -> int:
    if not edges:
        return 0

    rec = session.run(
        """
        UNWIND $edges AS e
        WITH e.src_key AS src_key, e.tgt_key AS tgt_key, max(e.score) AS score
        MATCH (a:Community {community_key: src_key})
        MATCH (b:Community {community_key: tgt_key})
        MERGE (a)-[r:COMM_SIMILAR_TO]->(b)
        SET r.sim_score = score
        RETURN count(DISTINCT src_key + '->' + tgt_key) AS pairs
        """,
        edges=edges,
    ).single()

    return int(rec["pairs"]) if rec and rec.get("pairs") is not None else 0


@dataclass
class BuildParams:
    top_k_neighbors: int = 20
    similarity_threshold: float = 0.78
    cross_doc_only: bool = True
    num_candidates: int = 200
    neo4j_edge_batch: int = 5000
    reset_edges: bool = False
    oversample_factor: int = 5
    bidirectional: bool = True 


def build_community_graph(
    cfg: PipelineConfig = DEFAULT_CONFIG,
    params: BuildParams = BuildParams(),
    only_doc_id: Optional[str] = None,
) -> None:
    es = build_es_client_from_cfg(cfg)
    if not es.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {cfg.es.url}")

    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    comm_index = community_index_name(cfg)

    try:
        with driver.session(database=cfg.neo4j.database) as session:
            ensure_neo4j_constraints(session)

            if params.reset_edges:
                logger.info("Resetting existing COMM_SIMILAR_TO edges...")
                reset_comm_sim_edges(session)

            comms = fetch_communities_for_doc(session, only_doc_id) if only_doc_id else fetch_all_communities(session)
            logger.info("Loaded %d communities from Neo4j%s", len(comms), f" (doc_id={only_doc_id})" if only_doc_id else "")

            total_edges = 0
            batch: List[Dict[str, Any]] = []

            k_fetch = params.top_k_neighbors * params.oversample_factor + 5

            for i, c in enumerate(comms, start=1):
                ck = c["community_key"]
                doc_id = c["doc_id"]

                qvec = es_get_community_vector(es, comm_index, ck)
                if qvec is None:
                    continue

                hits = es_knn_candidates(
                    es=es,
                    comm_index=comm_index,
                    query_vector=qvec,
                    k=k_fetch,
                    num_candidates=params.num_candidates,
                )

                a = np.array(qvec, dtype=np.float32)

                kept = 0
                for h in hits:
                    src = h.get("_source", {}) or {}
                    ck2 = src.get("community_key")
                    doc2 = src.get("doc_id")

                    if not ck2 or ck2 == ck:
                        continue
                    if params.cross_doc_only and doc2 == doc_id:
                        continue

                    v2 = src.get("vector")
                    if not isinstance(v2, list) or not v2:
                        continue

                    b = np.array(v2, dtype=np.float32)
                    sim = cosine(a, b)
                    if sim < params.similarity_threshold:
                        continue

                    batch.append({"src_key": ck, "tgt_key": ck2, "score": sim})
                    if params.bidirectional:
                        batch.append({"src_key": ck2, "tgt_key": ck, "score": sim})

                    kept += 1
                    if kept >= params.top_k_neighbors:
                        break

                if len(batch) >= params.neo4j_edge_batch:
                    written_pairs = write_comm_edges(session, batch)
                    total_edges += written_pairs
                    batch = []

                if i % 200 == 0:
                    logger.info("Processed %d/%d communities | edges so far: %d", i, len(comms), total_edges)

            if batch:
                written_pairs = write_comm_edges(session, batch)
                total_edges += written_pairs
                batch = []

            logger.info("Done. Created/updated COMM_SIMILAR_TO edges: %d", total_edges)

    finally:
        try:
            es.close()
        except Exception:
            pass
        driver.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--doc_id", type=str, default=None, help="Process only communities of this doc_id (incremental mode).")
    p.add_argument("--reset_edges", action="store_true", help="Delete all COMM_SIMILAR_TO edges before rebuilding.")
    p.add_argument("--no_bidir", action="store_true", help="Disable bidirectional edges.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    p = BuildParams(
        top_k_neighbors=DEFAULT_CONFIG.community_similarity.top_k_neighbors,
        similarity_threshold=DEFAULT_CONFIG.community_similarity.similarity_threshold,
        cross_doc_only=DEFAULT_CONFIG.community_similarity.cross_doc_only,
        reset_edges=bool(args.reset_edges),
        bidirectional=not bool(args.no_bidir),
    )
    build_community_graph(DEFAULT_CONFIG, p, only_doc_id=args.doc_id)
