import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = (SCRIPT_DIR / ".." / "elastic-start-local" / ".env").resolve()
load_dotenv(dotenv_path=ENV_PATH)

LLM_TABLES_JSONL = SCRIPT_DIR / "tables_llm.jsonl"
ROWS_JSONL       = SCRIPT_DIR / "rows_to_index.jsonl"

ES_URL      = os.getenv("ES_LOCAL_URL",      "http://localhost:9200")
ES_USER     = os.getenv("ES_LOCAL_USER",     "elastic")
ES_PASSWORD = os.getenv("ES_LOCAL_PASSWORD", "")

INDEX_INTROS     = "hybridqa_intros"
INDEX_TABLES_LLM = "hybridqa_tables_llm"
INDEX_ROWS       = "hybridqa_rows"

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMS   = 1024
EMBED_BATCH_SIZE = 64

RECREATE_INDICES = True
BULK_CHUNK_SIZE  = 100

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_load_json(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = safe_load_json(line)
            if obj:
                yield obj

def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("sentence-transformers not installed.") from e
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return model

def embed_texts(model, texts: List[str]) -> List[List[float]]:
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.extend(vecs.tolist())
    return all_vecs

def es_client() -> Elasticsearch:
    if ES_PASSWORD:
        return Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASSWORD), verify_certs=False)
    return Elasticsearch(ES_URL, verify_certs=False)

def base_index_settings() -> Dict[str, Any]:
    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "default": {"type": "standard"}
                }
            }
        },
        "mappings": {
            "dynamic": True,
            "properties": {
                "doc_id":        {"type": "keyword"},
                "table_id":      {"type": "keyword"},
                "serialization": {"type": "keyword"},
                "url":           {"type": "keyword"},
                "title":         {"type": "text"},
                "intro":         {"type": "text"},
                "section_title": {"type": "text"},
                "row_index":     {"type": "integer"},
                "to_index_text": {"type": "text"},
                "embedding": {
                    "type":       "dense_vector",
                    "dims":       EMBEDDING_DIMS,
                    "index":      True,
                    "similarity": "cosine",
                },
            },
        },
    }

def recreate_index(es: Elasticsearch, index_name: str) -> None:
    if es.indices.exists(index=index_name):
        if RECREATE_INDICES:
            es.indices.delete(index=index_name)
        else:
            return
    es.indices.create(index=index_name, **base_index_settings())

def bulk_index(
    es: Elasticsearch,
    index_name: str,
    records: List[Dict[str, Any]],
    id_field: str = "doc_id",
) -> Tuple[int, int]:
    ok = 0
    failed = 0
    actions = []
    for rec in records:
        _id = rec.get(id_field) or rec.get("table_id")
        if not _id:
            continue
        actions.append({
            "_op_type": "index",
            "_index":   index_name,
            "_id":      _id,
            "_source":  rec,
        })
        if len(actions) >= BULK_CHUNK_SIZE:
            success, errors = helpers.bulk(es, actions, raise_on_error=False)
            ok      += int(success)
            failed  += len(errors) if errors else 0
            actions  = []
    if actions:
        success, errors = helpers.bulk(es, actions, raise_on_error=False)
        ok     += int(success)
        failed += len(errors) if errors else 0
    return ok, failed

def build_intro_records(llm_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for obj in llm_tables:
        table_id = obj.get("table_id", "")
        if not table_id:
            continue
        intro = norm_ws(obj.get("intro", "") or "")
        records.append({
            "doc_id":        f"{table_id}__intro",
            "table_id":      table_id,
            "title":         obj.get("title", "") or "",
            "intro":         intro,
            "url":           obj.get("url", "") or "",
            "section_title": obj.get("section_title", "") or "",
            "serialization": "intro_only",
            "to_index_text": intro,
        })
    return records

def build_llm_records(llm_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for obj in llm_tables:
        table_id = obj.get("table_id", "")
        if not table_id:
            continue
        records.append({
            "doc_id":          f"{table_id}__llm",
            "table_id":        table_id,
            "title":           obj.get("title", "") or "",
            "intro":           obj.get("intro", "") or "",
            "lm_description":  obj.get("lm_description", "") or "",
            "url":             obj.get("url", "") or "",
            "section_title":   obj.get("section_title", "") or "",
            "serialization":   "llm_table",
            "to_index_text":   norm_ws(obj.get("to_index_text", "") or ""),
        })
    return records

def build_row_records(rows_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for obj in rows_data:
        table_id  = obj.get("table_id", "")
        row_index = obj.get("row_index", -1)
        doc_id    = obj.get("doc_id") or f"{table_id}__row_{row_index}"
        records.append({
            "doc_id":        doc_id,
            "table_id":      table_id,
            "row_index":     row_index,
            "title":         obj.get("title", "") or "",
            "url":           obj.get("url", "") or "",
            "section_title": obj.get("section_title", "") or "",
            "serialization": "row_level",
            "to_index_text": norm_ws(obj.get("to_index_text", "") or ""),
        })
    return records

def attach_embeddings(model, records: List[Dict[str, Any]]) -> None:
    texts = [r.get("to_index_text", "") or "" for r in records]
    vecs = embed_texts(model, texts)
    for rec, vec in zip(records, vecs):
        rec["embedding"] = vec

def main():
    model = load_embedder()
    es = es_client()

    for idx in [INDEX_INTROS, INDEX_TABLES_LLM, INDEX_ROWS]:
        recreate_index(es, idx)

    llm_tables = list(iter_jsonl(LLM_TABLES_JSONL))
    rows_data  = list(iter_jsonl(ROWS_JSONL))

    intro_records = build_intro_records(llm_tables)
    attach_embeddings(model, intro_records)
    bulk_index(es, INDEX_INTROS, intro_records)

    llm_records = build_llm_records(llm_tables)
    attach_embeddings(model, llm_records)
    bulk_index(es, INDEX_TABLES_LLM, llm_records)

    row_records = build_row_records(rows_data)
    attach_embeddings(model, row_records)
    bulk_index(es, INDEX_ROWS, row_records)

    for idx in [INDEX_INTROS, INDEX_TABLES_LLM, INDEX_ROWS]:
        es.indices.refresh(index=idx)

if __name__ == "__main__":
    main()