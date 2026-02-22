from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from datasets import load_dataset
from tqdm import tqdm

WIKI_API = "https://en.wikipedia.org/w/api.php"


def safe_id(s: str) -> str:
    s = re.sub(r"\s+", "-", (s or "").strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s[:80] if s else "x"


def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("hotpotqa_collector")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def fetch_wikipedia_page(
    session: requests.Session,
    title: str,
    fmt: str = "html",
    timeout: int = 30,
    max_retries: int = 5,
    backoff_base_s: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> str:
    headers = {"User-Agent": "HotpotQA-RAG-Collector/1.0"}

    if fmt == "wikitext":
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "formatversion": "2",
            "redirects": 1,
        }
    elif fmt == "plaintext":
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "exintro": 0,
            "explaintext": 1,
            "formatversion": "2",
            "redirects": 1,
        }
    else:
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "formatversion": "2",
            "redirects": 1,
        }

    for attempt in range(max_retries):
        try:
            r = session.get(WIKI_API, params=params, headers=headers, timeout=timeout)

            if r.status_code == 429:
                sleep_s = (2**attempt) * backoff_base_s
                if logger:
                    logger.warning(
                        f"429 rate limit su '{title}'. Retry {attempt+1}/{max_retries} in {sleep_s:.1f}s"
                    )
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            data = r.json()

            if fmt == "wikitext":
                page = (data.get("query", {}).get("pages") or [{}])[0]
                if page.get("missing"):
                    return ""
                rev = (page.get("revisions") or [{}])[0]
                return (rev.get("slots", {}).get("main", {}).get("content") or "")

            if fmt == "plaintext":
                page = (data.get("query", {}).get("pages") or [{}])[0]
                if page.get("missing"):
                    return ""
                return (page.get("extract") or "")

            parse = data.get("parse")
            if not parse:
                return ""
            html_fragment = parse.get("text") or ""
            return (
                "<!doctype html><html><head><meta charset='utf-8'></head>"
                f"<body>{html_fragment}</body></html>"
            )

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            sleep_s = (2**attempt) * backoff_base_s
            if logger:
                logger.warning(
                    f"Errore requests su '{title}' ({type(e).__name__}). "
                    f"Retry {attempt+1}/{max_retries} in {sleep_s:.1f}s"
                )
            time.sleep(sleep_s)

    return ""


def get_titles_from_hotpot(ex: Dict[str, Any], mode: str) -> List[str]:
    titles: List[str] = []

    if mode in ("support", "both"):
        sf = ex.get("supporting_facts", {})
        if isinstance(sf, dict):
            sf_titles = sf.get("title", [])
            if isinstance(sf_titles, list):
                titles.extend(sf_titles)
        elif isinstance(sf, list):
            for item in sf:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    titles.append(item[0])

    if mode in ("context", "both"):
        ctx = ex.get("context", {})
        if isinstance(ctx, dict):
            ctx_titles = ctx.get("title", [])
            if isinstance(ctx_titles, list):
                titles.extend(ctx_titles)
        elif isinstance(ctx, list):
            for item in ctx:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    titles.append(item[0])

    seen: Set[str] = set()
    out: List[str] = []
    for t in titles:
        if isinstance(t, str):
            t = t.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


def stratified_sample(dataset, max_samples: int = 400, seed: int = 42) -> List[int]:
    random.seed(seed)
    strata: Dict[Tuple[str, str], List[int]] = {}

    for idx, ex in enumerate(dataset):
        level = ex.get("level", "unknown")
        qtype = ex.get("type", "unknown")
        key = (level, qtype)
        strata.setdefault(key, []).append(idx)

    total = len(dataset)
    sampled_indices: List[int] = []

    for key, indices in strata.items():
        stratum_size = len(indices)
        n_sample = max(1, int(max_samples * stratum_size / total))
        n_sample = min(n_sample, stratum_size)
        sampled_indices.extend(random.sample(indices, n_sample))

    if len(sampled_indices) < max_samples:
        remaining = max_samples - len(sampled_indices)
        all_indices = set(range(len(dataset)))
        not_sampled = list(all_indices - set(sampled_indices))
        if not_sampled:
            sampled_indices.extend(random.sample(not_sampled, min(remaining, len(not_sampled))))
    elif len(sampled_indices) > max_samples:
        sampled_indices = random.sample(sampled_indices, max_samples)

    return sorted(sampled_indices)


@dataclass
class Stats:
    examples_seen: int = 0
    examples_written: int = 0
    examples_skipped_empty_qa: int = 0
    pages_attempted: int = 0
    pages_saved: int = 0
    pages_empty: int = 0
    pages_failed: int = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--titles", default="both", choices=["support", "context", "both"])
    ap.add_argument("--wiki_format", default="html", choices=["html", "plaintext", "wikitext"])
    ap.add_argument("--max_examples", type=int, default=400)
    ap.add_argument("--max_pages_per_q", type=int, default=4)
    ap.add_argument("--sleep_s", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_jsonl", type=Path, default=Path("hotpotqa_records.jsonl"))
    ap.add_argument("--out_evidence_dir", type=Path, default=Path("hotpotqa_wiki_full"))
    ap.add_argument("--stratify", action="store_true", default=True)

    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--debug_trace", action="store_true", help="Include traceback completo nei log (più verboso)")
    ap.add_argument("--errors_jsonl", type=Path, default=None, help="Salva errori in JSONL (uno per riga)")
    ap.add_argument("--timeout_s", type=int, default=30)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--backoff_base_s", type=float, default=1.0)
    args = ap.parse_args()

    logger = setup_logger(args.log_level)
    logger.info(f"Loading dataset hotpot_qa/fullwiki split={args.split} ...")
    ds = load_dataset("hotpot_qa", "fullwiki", split=args.split)

    if args.stratify:
        selected_indices = stratified_sample(ds, max_samples=args.max_examples, seed=args.seed)
        logger.info(f"Selected {len(selected_indices)} indices (stratified)")
    else:
        selected_indices = list(range(min(args.max_examples, len(ds))))
        logger.info(f"Selected {len(selected_indices)} indices (first N)")

    args.out_evidence_dir.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.errors_jsonl:
        args.errors_jsonl.parent.mkdir(parents=True, exist_ok=True)

    ext = {"html": "html", "plaintext": "txt", "wikitext": "wiki"}[args.wiki_format]

    stats = Stats()
    failures_by_exc = Counter()

    err_fout = None
    if args.errors_jsonl:
        err_fout = args.errors_jsonl.open("w", encoding="utf-8")

    with requests.Session() as session, args.out_jsonl.open("w", encoding="utf-8") as fout:
        pbar = tqdm(selected_indices, desc="Examples", unit="ex")

        for original_idx in pbar:
            stats.examples_seen += 1

            ex = ds[original_idx]
            q = (ex.get("question") or "").strip()
            a = (ex.get("answer") or "").strip()
            if not q or not a:
                stats.examples_skipped_empty_qa += 1
                continue

            level = ex.get("level", "unknown")
            qtype = ex.get("type", "unknown")
            rec_id = f"HotpotQA-{safe_id(args.split)}-{original_idx:08d}"
            rec_dir = args.out_evidence_dir / rec_id
            rec_dir.mkdir(parents=True, exist_ok=True)

            titles = get_titles_from_hotpot(ex, args.titles)[: args.max_pages_per_q]
            raw_files: List[str] = []

            sub = tqdm(
                titles,
                desc=f"Pages [{rec_id}]",
                unit="page",
                leave=False,
            )

            for j, title in enumerate(sub):
                stats.pages_attempted += 1
                try:
                    content = fetch_wikipedia_page(
                        session=session,
                        title=title,
                        fmt=args.wiki_format,
                        timeout=args.timeout_s,
                        max_retries=args.max_retries,
                        backoff_base_s=args.backoff_base_s,
                        logger=logger if args.log_level == "DEBUG" else None,
                    )

                    if not content.strip():
                        stats.pages_empty += 1
                        continue

                    fpath = rec_dir / f"{j:02d}_{safe_id(title)}.{ext}"
                    fpath.write_text(content, encoding="utf-8")
                    raw_files.append(fpath.as_posix())
                    stats.pages_saved += 1

                    if args.sleep_s:
                        time.sleep(args.sleep_s)

                except Exception as e:
                    stats.pages_failed += 1
                    failures_by_exc[type(e).__name__] += 1

                    msg = f"Failed page title='{title}' idx={original_idx} rec_id={rec_id} exc={type(e).__name__}"
                    if args.debug_trace:
                        logger.exception(msg)
                    else:
                        logger.warning(msg + f" ({e})")

                    if err_fout is not None:
                        err_rec = {
                            "rec_id": rec_id,
                            "dataset_idx": original_idx,
                            "title": title,
                            "exception": type(e).__name__,
                            "message": str(e),
                        }
                        err_fout.write(json.dumps(err_rec, ensure_ascii=False) + "\n")

                    continue

            record = {
                "id": rec_id,
                "source": rec_dir.as_posix(),
                "question": q,
                "answer": a,
                "question_type": qtype,
                "level": level,
                "raw_evidence_files": raw_files,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.examples_written += 1

            pbar.set_postfix(
                {
                    "written": stats.examples_written,
                    "p_saved": stats.pages_saved,
                    "p_fail": stats.pages_failed,
                    "p_empty": stats.pages_empty,
                }
            )

    if err_fout is not None:
        err_fout.close()

    logger.info("Done.")
    logger.info(
        "Examples: seen=%d written=%d skipped_empty_qa=%d",
        stats.examples_seen,
        stats.examples_written,
        stats.examples_skipped_empty_qa,
    )
    logger.info(
        "Pages: attempted=%d saved=%d empty=%d failed=%d",
        stats.pages_attempted,
        stats.pages_saved,
        stats.pages_empty,
        stats.pages_failed,
    )
    if failures_by_exc:
        top = ", ".join(f"{k}:{v}" for k, v in failures_by_exc.most_common(10))
        logger.info("Top page failure exceptions: %s", top)


if __name__ == "__main__":
    main()
