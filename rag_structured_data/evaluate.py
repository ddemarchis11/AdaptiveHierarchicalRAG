import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = (SCRIPT_DIR / ".." / "elastic-start-local" / ".env").resolve()
if not ENV_PATH.exists():
    raise FileNotFoundError(f"Missing .env at: {ENV_PATH}")
load_dotenv(dotenv_path=ENV_PATH)

QAS_FILE   = SCRIPT_DIR / "subset_qas_only.jsonl"
RESULTS_MD = SCRIPT_DIR / "results.md"

ES_URL      = os.getenv("ES_LOCAL_URL",      "http://localhost:9200")
ES_USER     = os.getenv("ES_LOCAL_USER",     "elastic")
ES_PASSWORD = os.getenv("ES_LOCAL_PASSWORD", "")

INDEX_TABLES_LLM = "hybridqa_tables_llm"
INDEX_ROWS       = "hybridqa_rows"
INDEX_INTROS     = "hybridqa_intros"

HIT_KS = [1, 3, 5, 10, 20]

TOPK_SPARSE = 100
TOPK_DENSE  = 100
RRF_K       = 60

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMS   = 1024


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def safe_load_json(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = safe_load_json(line)
            if obj:
                yield obj


def es_client() -> Elasticsearch:
    if ES_PASSWORD:
        return Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASSWORD), verify_certs=False)
    return Elasticsearch(ES_URL, verify_certs=False)


def bm25_search(es: Elasticsearch, index: str, query: str, topk: int) -> List[Dict[str, Any]]:
    q = norm_ws(query)
    body = {
        "size": topk,
        "_source": ["doc_id", "table_id"],
        "query": {
            "bool": {
                "should": [
                    {"match": {"to_index_text": {"query": q, "boost": 3.0}}},
                    {"match": {"title":         {"query": q, "boost": 1.5}}},
                    {"match": {"intro":          {"query": q, "boost": 1.0}}},
                ],
                "minimum_should_match": 1,
            }
        },
    }
    resp = es.search(index=index, body=body)
    return _parse_hits(resp)


def dense_search(es: Elasticsearch, index: str, query_vec: List[float], topk: int) -> List[Dict[str, Any]]:
    body = {
        "size": topk,
        "_source": ["doc_id", "table_id"],
        "knn": {
            "field":          "embedding",
            "query_vector":   query_vec,
            "k":              topk,
            "num_candidates": max(200, topk * 4),
        },
    }
    resp = es.search(index=index, body=body)
    return _parse_hits(resp)


def _parse_hits(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {}) or {}
        out.append({
            "doc_id":   h.get("_id") or src.get("doc_id"),
            "table_id": src.get("table_id"),
            "score":    float(h.get("_score") or 0.0),
        })
    return out


def rrf_fuse(
    sparse: List[Dict[str, Any]],
    dense:  List[Dict[str, Any]],
    rrf_k:  int,
    topn:   int,
) -> List[Dict[str, Any]]:
    scores: Dict[str, float]       = {}
    meta:   Dict[str, Dict[str, Any]] = {}

    for rank, d in enumerate(sparse, start=1):
        did = d["doc_id"]
        if not did:
            continue
        scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rank)
        meta.setdefault(did, {}).update(d)

    for rank, d in enumerate(dense, start=1):
        did = d["doc_id"]
        if not did:
            continue
        scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rank)
        meta.setdefault(did, {}).update(d)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    return [
        {"doc_id": did, "table_id": meta[did].get("table_id"), "score": sc, "rank": i}
        for i, (did, sc) in enumerate(ranked, start=1)
    ]

def compute_metrics(
    retrieved: List[Dict[str, Any]], gold_table_id: str
) -> Tuple[Dict[int, int], float]:
    hits = {k: 0 for k in HIT_KS}
    mrr  = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d.get("table_id") == gold_table_id:
            mrr = 1.0 / i
            for k in HIT_KS:
                if i <= k:
                    hits[k] = 1
            break
    return hits, mrr


def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer 
        print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
        model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Model loaded.")
        return model
    except Exception as e:
        raise RuntimeError(f"Cannot load embedder: {e}") from e


def load_queries() -> List[Dict[str, Any]]:
    if not QAS_FILE.exists():
        raise FileNotFoundError(f"Missing qas file: {QAS_FILE}")
    qas = []
    for obj in iter_jsonl(QAS_FILE):
        if obj.get("table_only") is not True:
            continue
        q        = obj.get("question") or ""
        table_id = obj.get("table_id") or ""
        if not q or not table_id:
            continue
        qas.append({
            "question_id": obj.get("question_id"),
            "question":    q,
            "table_id":    table_id,
        })
    return qas

@dataclass
class EvalRunResult:
    label:         str  
    index_name:    str
    mode:          str  
    n_queries:     int
    hit_at:        Dict[int, float]
    mrr:           float
    avg_latency_ms: float


def evaluate_index(
    es:         Elasticsearch,
    index_name: str,
    qas:        List[Dict[str, Any]],
    embedder,
) -> List[EvalRunResult]:
    modes = ["sparse", "dense", "hybrid"]
    hit_sums   = {m: {k: 0   for k in HIT_KS} for m in modes}
    mrr_sums   = {m: 0.0                        for m in modes}
    lat_sums   = {m: 0.0                        for m in modes}

    for qa in tqdm(qas, desc=f"{index_name}", unit="q", leave=True):
        q        = qa["question"]
        gold_tid = qa["table_id"]

        vec = embedder.encode([q], normalize_embeddings=True)[0].tolist()
        if len(vec) != EMBEDDING_DIMS:
            raise ValueError(f"query embedding dims {len(vec)} != {EMBEDDING_DIMS}")

        t0     = time.time()
        sparse = bm25_search(es, index_name, q, TOPK_SPARSE)
        lat_sums["sparse"] += (time.time() - t0) * 1000.0
        h, m = compute_metrics(sparse, gold_tid)
        for k in HIT_KS:
            hit_sums["sparse"][k] += h[k]
        mrr_sums["sparse"] += m

        t0    = time.time()
        dense = dense_search(es, index_name, vec, TOPK_DENSE)
        lat_sums["dense"] += (time.time() - t0) * 1000.0
        h, m = compute_metrics(dense, gold_tid)
        for k in HIT_KS:
            hit_sums["dense"][k] += h[k]
        mrr_sums["dense"] += m

        t0     = time.time()
        hybrid = rrf_fuse(sparse, dense, rrf_k=RRF_K, topn=max(HIT_KS))
        lat_sums["hybrid"] += (time.time() - t0) * 1000.0
        h, m = compute_metrics(hybrid, gold_tid)
        for k in HIT_KS:
            hit_sums["hybrid"][k] += h[k]
        mrr_sums["hybrid"] += m

    n = max(1, len(qas))
    results = []
    for mode in modes:
        results.append(EvalRunResult(
            label          = f"{index_name} / {mode}",
            index_name     = index_name,
            mode           = mode,
            n_queries      = len(qas),
            hit_at         = {k: hit_sums[mode][k] / n for k in HIT_KS},
            mrr            = mrr_sums[mode] / n,
            avg_latency_ms = lat_sums[mode] / n,
        ))
    return results

def md_table(results: List[EvalRunResult]) -> str:
    cols  = ["Index", "Mode", "N", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "MRR", "Avg Lat (ms)"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for r in results:
        lines.append(
            "| " + " | ".join([
                r.index_name,
                r.mode,
                str(r.n_queries),
                f"{r.hit_at[1]:.3f}",
                f"{r.hit_at[3]:.3f}",
                f"{r.hit_at[5]:.3f}",
                f"{r.hit_at[10]:.3f}",
                f"{r.hit_at[20]:.3f}",
                f"{r.mrr:.3f}",
                f"{r.avg_latency_ms:.1f}",
            ]) + " |"
        )
    return "\n".join(lines)


def write_results_md(results: List[EvalRunResult]) -> None:
    ts    = time.strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        "# Retrieval evaluation (Hit@k)\n",
        f"- Timestamp: `{ts}`",
        f"- QAs file: `{QAS_FILE.name}` (filtered: `table_only == true`)",
        f"- ES URL: `{ES_URL}`",
        f"- TOPK_SPARSE: `{TOPK_SPARSE}`, TOPK_DENSE: `{TOPK_DENSE}`, RRF_K: `{RRF_K}`",
        f"- Dense model: `{EMBED_MODEL_NAME}` (dims={EMBEDDING_DIMS})",
        "",
        "## Summary\n",
        md_table(results),
        "",
        "## Notes\n",
        "- Hybrid uses Reciprocal Rank Fusion (RRF) of sparse and dense results.",
        "- Latency for hybrid is the RRF fusion overhead only (sparse + dense already measured separately).",
        "",
    ]
    RESULTS_MD.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved report -> {RESULTS_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    qas = load_queries()
    if not qas:
        raise RuntimeError(f"No table_only queries found in {QAS_FILE}")
    print(f"Loaded {len(qas)} queries.")

    es = es_client()
    indices = [INDEX_TABLES_LLM, INDEX_ROWS, INDEX_INTROS]
    for idx in indices:
        if not es.indices.exists(index=idx):
            raise FileNotFoundError(f"Missing ES index: {idx}")

    embedder = load_embedder()

    all_results: List[EvalRunResult] = []
    for idx in tqdm(indices, desc="Indices", unit="idx", leave=True):
        results = evaluate_index(es, idx, qas, embedder)
        all_results.extend(results)

    print("\n=== Retrieval evaluation (Hit@k) ===")
    print(f"QAs file: {QAS_FILE.name}")
    print(md_table(all_results))

    write_results_md(all_results)


if __name__ == "__main__":
    main()