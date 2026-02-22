from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Tuple, Optional

from neo4j import GraphDatabase, Session
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from elasticsearch import Elasticsearch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEFAULT_CONFIG, PipelineConfig, LLMConfig
from llm_stub import get_llm_client 

logger = logging.getLogger(__name__)

community_summary_system = """
You are an expert technical writer assisting with a GraphRAG system.
Your task is to analyze a collection of text chunks that belong to the same semantic community (topic).

Output Format:
1. **Title**: A short, concise title for this topic (max 5-7 words).
2. **Summary**: A comprehensive synthesis of the key points discussed in these chunks (3-5 sentences).
3. **Keywords**: A list of 3-5 important tags/keywords.

Constraints:
- Do NOT output preamble or conversational filler.
- Focus on the shared theme connecting these chunks.
- If the chunks are unrelated (noise), title it "Miscellaneous" and provide a brief description.
"""

community_summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", community_summary_system),
        ("user", "Chunks:\n{passages}\n\nGenerate the community report."),
    ]
)


def build_es_client_from_cfg(cfg: PipelineConfig) -> Elasticsearch:
    user = getattr(cfg.es, "user", None)
    pwd = getattr(cfg.es, "password", None)
    basic_auth = (user, pwd) if (user and pwd) else None
    return Elasticsearch(
        cfg.es.url,
        basic_auth=basic_auth,
        verify_certs=getattr(cfg.es, "verify_certs", True),
    )


def get_community_index_name(cfg: PipelineConfig) -> str:
    explicit = getattr(cfg.es, "community_index_name", None)
    return explicit or f"{cfg.es.index_name}_communities"


def create_community_index_if_not_exists(es: Elasticsearch, index_name: str, embedding_dim: int):
    if es.indices.exists(index=index_name):
        return

    mapping = {
        "mappings": {
            "properties": {
                "community_key": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "community_id": {"type": "integer"},
                "title": {"type": "text"},
                "summary": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": int(embedding_dim),
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    }
    es.indices.create(index=index_name, body=mapping)
    logger.info("Created ES community index '%s' (dim=%d)", index_name, embedding_dim)


def write_pagerank_for_doc(session: Session, doc_id: str, concurrency: int = 4) -> None:
    gname = f"pr_doc_{hash(doc_id) & 0x7fffffff}"
    session.run(
        """
        CALL gds.graph.project.cypher(
          $gname,
          'MATCH (c:Chunk {doc_id: $doc_id}) RETURN id(c) AS id',
          'MATCH (c1:Chunk {doc_id: $doc_id})-[r:SIMILAR_TO|NEXT]->(c2:Chunk {doc_id: $doc_id})
           RETURN id(c1) AS source,
                  id(c2) AS target,
                  CASE WHEN type(r) = "SIMILAR_TO" THEN r.sim_score ELSE 0.1 END AS sim_score',
          {parameters: {doc_id: $doc_id}, validateRelationships: false}
        )
        """,
        gname=gname,
        doc_id=doc_id,
    )

    session.run(
        """
        CALL gds.pageRank.write($gname, {
            writeProperty: 'pagerank',
            relationshipWeightProperty: 'sim_score',
            concurrency: $concurrency
        })
        """,
        gname=gname,
        concurrency=int(concurrency),
    )

    session.run("CALL gds.graph.drop($gname)", gname=gname)


def write_communities_for_doc(session: Session, doc_id: str, algo: str, concurrency: int = 4, next_weight: float = 0.1) -> Tuple[int, float]:
    algo = algo.lower()
    if algo not in {"louvain", "leiden"}:
        raise ValueError("community_algo must be 'louvain' or 'leiden'")

    gname = f"comm_doc_{hash(doc_id) & 0x7fffffff}"
    session.run(
        """
        CALL gds.graph.project.cypher(
          $gname,
          'MATCH (c:Chunk {doc_id: $doc_id}) RETURN id(c) AS id',
          'MATCH (c1:Chunk {doc_id: $doc_id})-[r:SIMILAR_TO|NEXT]->(c2:Chunk {doc_id: $doc_id})
           RETURN id(c1) AS source,
                  id(c2) AS target,
                  CASE WHEN type(r) = "SIMILAR_TO" THEN r.sim_score ELSE $next_weight END AS sim_score',
          {parameters: {doc_id: $doc_id, next_weight: $next_weight}, validateRelationships: false}
        )
        """,
        gname=gname,
        doc_id=doc_id,
        next_weight=float(next_weight),
    )

    rec = session.run(
        f"""
        CALL gds.{algo}.write($gname, {{
            writeProperty: 'community_id',
            relationshipWeightProperty: 'sim_score',
            includeIntermediateCommunities: false,
            concurrency: $concurrency
        }})
        YIELD communityCount, modularity
        RETURN communityCount AS communityCount, modularity AS modularity
        """,
        gname=gname,
        concurrency=int(concurrency),
    ).single()

    session.run("CALL gds.graph.drop($gname)", gname=gname)

    cc = int(rec["communityCount"]) if rec else 0
    mod = float(rec["modularity"]) if rec else 0.0
    return cc, mod


def fetch_top_chunks_for_community(
    session: Session,
    doc_id: str,
    community_id: int,
    limit: int,
) -> List[Tuple[str, str]]:
    query = """
    MATCH (c:Chunk)
    WHERE c.doc_id = $doc_id AND c.community_id = $cid
    RETURN c.doc_id AS doc_id, c.text AS text
    ORDER BY c.pagerank DESC
    LIMIT $limit
    """
    result = session.run(query, doc_id=doc_id, cid=int(community_id), limit=int(limit))
    return [(record["doc_id"], record["text"]) for record in result]


def _strip_think_tags(raw: str) -> str:
    start = raw.find("<think>")
    end = raw.find("</think>")
    if start != -1 and end != -1 and end > start:
        return (raw[:start] + raw[end + len("</think>"):]).strip()
    return raw.strip()


def summarize_community(
    session: Session,
    es: Elasticsearch,
    model: SentenceTransformer,
    doc_id: str,
    community_id: int,
    cfg: PipelineConfig,
    llm,
    top_n_chunks: int = 5,
) -> None:
    passages = fetch_top_chunks_for_community(session, doc_id=doc_id, community_id=community_id, limit=top_n_chunks)
    if not passages:
        return

    passages_text = "\n\n".join(f"[{doc_id}] {text}" for doc_id, text in passages)

    try:
        messages = community_summary_prompt.format_messages(passages=passages_text)
        response = llm.invoke(messages)
        raw = getattr(response, "content", "") or ""
        summary_block = _strip_think_tags(raw) or "Summary generation failed."
    except Exception as e:
        logger.error("Error summarizing community %s in doc_id=%s: %s", community_id, doc_id, e)
        summary_block = "Summary generation failed."

    title = f"Topic {community_id}"
    emb = model.encode(summary_block, convert_to_numpy=True, normalize_embeddings=True).tolist()
    community_key = f"{doc_id}_{community_id}"

    cypher_update = """
    MERGE (comm:Community {community_key: $community_key})
    SET comm.summary      = $summary,
        comm.title        = $title,
        comm.doc_id       = $doc_id,
        comm.community_id = $community_id

    WITH comm
    MATCH (c:Chunk)
    WHERE c.doc_id = $doc_id AND c.community_id = $community_id
    MERGE (comm)-[:HAS_CHUNK]->(c)
    """
    session.run(
        cypher_update,
        community_key=community_key,
        summary=summary_block,
        title=title,
        doc_id=doc_id,
        community_id=int(community_id),
    )

    comm_index = get_community_index_name(cfg)
    es.index(
        index=comm_index,
        id=f"community_{community_key}",
        document={
            "community_key": community_key,
            "doc_id": doc_id,
            "community_id": int(community_id),
            "title": title,
            "summary": summary_block,
            "vector": emb,
        },
        refresh=False,
    )


def compute_communities_and_summaries(
    config: PipelineConfig = DEFAULT_CONFIG,
    top_n_chunks: int = 10,
    doc_id: Optional[str] = None,
) -> None:
    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password),
    )

    es = build_es_client_from_cfg(config)
    if not es.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {config.es.url}")

    logger.info("Loading embedding model: %s", config.embeddings.model_name)
    model = SentenceTransformer(config.embeddings.model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    comm_index = get_community_index_name(config)
    create_community_index_if_not_exists(es, comm_index, embedding_dim)

    config.llm = LLMConfig(
        api_key=config.llm.api_key,
        model=config.graph.community_model,
        max_tokens=512,
    )
    llm = get_llm_client(config)

    algo = config.graph.community_algo
    next_weight = float(getattr(getattr(config, "graph", object()), "next_weight", 0.1))

    try:
        with driver.session(database=config.neo4j.database) as session:
            if doc_id:
                logger.info("Incremental mode: computing communities for doc_id=%s", doc_id)

                write_pagerank_for_doc(session, doc_id, concurrency=config.graph.gds_concurrency)
                cc, mod = write_communities_for_doc(
                    session,
                    doc_id=doc_id,
                    algo=algo,
                    concurrency=config.graph.gds_concurrency,
                    next_weight=next_weight,
                )
                logger.info("%s for doc_id=%s: communityCount=%d, modularity=%.4f", algo.title(), doc_id, cc, mod)

                rows = session.run(
                    """
                    MATCH (c:Chunk {doc_id: $doc_id})
                    WHERE c.community_id IS NOT NULL
                    RETURN DISTINCT c.community_id AS cid
                    """,
                    doc_id=doc_id,
                )
                cids = [int(r["cid"]) for r in rows]

                logger.info("Summarizing %d communities for doc_id=%s", len(cids), doc_id)
                for cid in cids:
                    summarize_community(
                        session=session,
                        es=es,
                        model=model,
                        doc_id=doc_id,
                        community_id=cid,
                        cfg=config,
                        llm=llm,
                        top_n_chunks=top_n_chunks,
                    )
            else:
                rows = session.run("MATCH (c:Chunk) RETURN DISTINCT c.doc_id AS doc_id")
                doc_ids = [r["doc_id"] for r in rows]
                logger.info("Batch mode: computing communities per-document on %d documents...", len(doc_ids))

                for did in doc_ids:
                    write_pagerank_for_doc(session, did, concurrency=config.graph.gds_concurrency)
                    cc, mod = write_communities_for_doc(
                        session,
                        doc_id=did,
                        algo=algo,
                        concurrency=config.graph.gds_concurrency,
                        next_weight=next_weight,
                    )
                    logger.info("%s for doc_id=%s: communityCount=%d, modularity=%.4f", algo.title(), did, cc, mod)

                    rows2 = session.run(
                        """
                        MATCH (c:Chunk {doc_id: $doc_id})
                        WHERE c.community_id IS NOT NULL
                        RETURN DISTINCT c.community_id AS cid
                        """,
                        doc_id=did,
                    )
                    cids = [int(r["cid"]) for r in rows2]
                    for cid in cids:
                        summarize_community(
                            session=session,
                            es=es,
                            model=model,
                            doc_id=did,
                            community_id=cid,
                            cfg=config,
                            llm=llm,
                            top_n_chunks=top_n_chunks,
                        )

    finally:
        try:
            es.close()
        except Exception:
            pass
        driver.close()

    logger.info("Computed communities and summaries successfully.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--doc_id", type=str, default=None, help="Compute communities only for this doc_id (incremental mode).")
    p.add_argument("--top_n_chunks", type=int, default=DEFAULT_CONFIG.graph.top_n_chunks, help="Chunks used to summarize each community.")
    return p.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    args = _parse_args()
    compute_communities_and_summaries(DEFAULT_CONFIG, top_n_chunks=int(args.top_n_chunks), doc_id=args.doc_id)
