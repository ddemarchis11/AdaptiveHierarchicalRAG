from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from neo4j import GraphDatabase, Session
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEFAULT_CONFIG, PipelineConfig 

logger = logging.getLogger(__name__)


def iter_documents(path: Path) -> Iterable[tuple[str, str]]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = data.get("context") or data.get("text") or ""
            if not text:
                continue

            doc_id = data.get("corpus_name")
            if doc_id is None:
                doc_id = data.get("id") or f"doc_{i}"

            yield str(doc_id), text


def chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tokens,
        chunk_overlap=overlap_tokens,
        encoding_name="cl100k_base",
    )
    return splitter.split_text(text)


def upsert_chunks_neo4j(session: Session, chunk_records: List[Dict[str, Any]]):
    if not chunk_records:
        return

    query = """
    UNWIND $rows AS row
    MERGE (c:Chunk {chunk_id: row.chunk_id})
      ON CREATE SET c.doc_id = row.doc_id,
                    c.position = row.position,
                    c.text = row.text
      ON MATCH  SET c.doc_id = row.doc_id,
                    c.position = row.position,
                    c.text = row.text
    """
    session.run(query, rows=chunk_records)


def ingest_chunks_neo4j_only(config: PipelineConfig = DEFAULT_CONFIG, only_doc_id: Optional[str] = None):
    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password),
    )

    chunk_size = int(config.chunking.chunk_size)
    overlap = int(chunk_size * float(config.chunking.overlap_ratio))
    batch_size = int(getattr(config, "neo4j_chunk_upsert_batch", 1000))

    total_chunks = 0
    total_docs = 0

    try:
        with driver.session(database=config.neo4j.database) as session:
            batch: List[Dict[str, Any]] = []

            for doc_id, text in iter_documents(config.documents_path):
                if only_doc_id and doc_id != only_doc_id:
                    continue

                total_docs += 1
                chunks = chunk_text(text, chunk_size, overlap)

                for idx, chunk_text_val in enumerate(chunks):
                    chunk_id = f"{doc_id}_c_{idx}"
                    batch.append(
                        {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "position": idx,
                            "text": chunk_text_val,
                        }
                    )
                    total_chunks += 1

                    if len(batch) >= batch_size:
                        upsert_chunks_neo4j(session, batch)
                        batch = []

            if batch:
                upsert_chunks_neo4j(session, batch)

        logger.info("Neo4j upsert complete. Docs: %d | Chunks: %d", total_docs, total_chunks)

    finally:
        driver.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--doc_id", type=str, default=None, help="Upsert only this doc_id from the JSONL file.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()
    ingest_chunks_neo4j_only(DEFAULT_CONFIG, only_doc_id=args.doc_id)
