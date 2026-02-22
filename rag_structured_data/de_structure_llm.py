import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))  
from config import DEFAULT_CONFIG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "subset_tables_only.jsonl")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "tables_llm.jsonl")
CACHE_DIR = os.path.join(SCRIPT_DIR, "lm_cache")

MAX_ROWS = 20
CHUNK_SIZE = 10
FORCE = False

PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from llm_stub import get_llm_client  # noqa: E402


def safe_load_json(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def row_to_kv(header: List[str], row: List[Any]) -> str:
    parts = []
    for h, v in zip(header, row):
        h2 = normalize_ws(str(h))
        v2 = normalize_ws(str(v))
        if h2 and v2:
            parts.append(f"{h2}: {v2}")
    return "; ".join(parts)


def select_rows_informative(table: Dict[str, Any], max_rows: int) -> List[Tuple[int, List[Any]]]:
    rows = table.get("rows", []) or []
    header = [normalize_ws(str(h)) for h in (table.get("header", []) or [])]
    n = len(rows)
    if n == 0:
        return []
    if max_rows <= 0 or n <= max_rows:
        return [(i, rows[i]) for i in range(n)]

    scored = []
    for i, r in enumerate(rows):
        kv = row_to_kv(header, r)
        score = len(kv)
        score += 10 * len(re.findall(r"\d", kv))
        score += 5 * len(re.findall(r"\b(19|20)\d{2}\b", kv))
        scored.append((score, i))
    scored.sort(reverse=True)
    idxs = sorted([i for _, i in scored[:max_rows]])
    return [(i, rows[i]) for i in idxs]


def chunk_rows(selected: List[Tuple[int, List[Any]]], chunk_size: int) -> List[List[Tuple[int, List[Any]]]]:
    if chunk_size <= 0:
        return [selected]
    return [selected[i : i + chunk_size] for i in range(0, len(selected), chunk_size)]


def build_prompt(table: Dict[str, Any]) -> List[Dict[str, str]]:
    title = normalize_ws(table.get("title", ""))
    intro = normalize_ws(table.get("intro", ""))
    section_title = normalize_ws(table.get("section_title", ""))

    header = [normalize_ws(str(h)) for h in (table.get("header", []) or [])]
    header_s = ", ".join([h for h in header if h])

    selected = select_rows_informative(table, max_rows=MAX_ROWS)
    blocks = chunk_rows(selected, chunk_size=CHUNK_SIZE)

    block_texts = []
    for b in blocks:
        lines = []
        for (i, row) in b:
            kv = row_to_kv(header, row)
            if kv:
                lines.append(f"row {i+1}: {kv}")
        if lines:
            block_texts.append("\n".join(lines))

    n_rows = (table.get("meta") or {}).get("n_rows", len(table.get("rows", []) or []))
    n_cols = (table.get("meta") or {}).get("n_cols", len(table.get("header", []) or []))

    system = (
        "You rewrite tables into natural-language descriptions optimized for semantic retrieval.\n"
        "You MUST describe the TABLE content (not the page, not the structure).\n"
        "You may use the intro only as context for wording, but do not summarize the intro.\n"
        "Do NOT invent facts/values; use only the provided rows/columns.\n"
        "Write compact, information-dense plain text.\n"
        "No JSON. No headings. No bullet lists.\n"
        "IMPORTANT: do NOT describe the table structure (columns, rows, format). "
        "Instead, EXPRESS the relationships between cell values in natural language sentences. "
        "For each entry or group of entries, fuse the relevant column values into meaningful statements "
        "(e.g. 'Chuck Jones directed For Scent-imental Reasons in 1949 for the Merrie Melodies series, co-starring Penelope'). "
        "Every specific value (name, date, title, figure) must appear explicitly in the text.\n"
    )    
    
    user_parts = []
    if title:
        user_parts.append(f"Title: {title}")
    if section_title:
        user_parts.append(f"Section: {section_title}")
    if intro:
        user_parts.append(f"Intro (context only): {intro}")
    if header_s:
        user_parts.append(f"Columns: {header_s}")
    user_parts.append(f"Table size: {n_rows} rows, {n_cols} columns.")
    user_parts.append("Table rows/blocks:")
    for i, btxt in enumerate(block_texts, start=1):
        user_parts.append(f"[Block {i}]\n{btxt}")

    user_parts.append(
        "Task: Write a flowing prose description (max ~200 words) that interprets the table content. "
        "For each row or group of related rows, produce a sentence that fuses the column values into a meaningful statement expressing their relationships. "
        "Do not describe the table as an object ('this table contains...', 'each row has...'). "
        "Do not use aggregates or statistics as a substitute for specific values. "
        "Write as if narrating the content to someone who cannot see the table."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def cache_key(table_id: str, model_name: str) -> str:
    raw = f"{table_id}|{model_name}|{MAX_ROWS}|{CHUNK_SIZE}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def call_llm(llm, messages: List[Dict[str, str]]) -> str:
    resp = llm.invoke(messages)
    return normalize_ws(getattr(resp, "content", ""))


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")

    llm = get_llm_client(DEFAULT_CONFIG)
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"

    written = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            table = safe_load_json(line)
            if not table:
                continue

            table_id = table.get("table_id")
            if not table_id:
                continue

            ck = cache_key(table_id, str(model_name))
            cache_path = os.path.join(CACHE_DIR, f"{table_id}.{ck}.txt")

            if os.path.exists(cache_path) and not FORCE:
                with open(cache_path, "r", encoding="utf-8") as f:
                    lm_desc = normalize_ws(f.read())
            else:
                messages = build_prompt(table)
                lm_desc = call_llm(llm, messages)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(lm_desc)

            title = table.get("title", "") or ""
            intro = table.get("intro", "") or ""

            to_index_text = "\n".join(
                [normalize_ws(title), normalize_ws(intro), normalize_ws(lm_desc)]
            ).strip()

            out_obj = {
                "table_id": table_id,
                "title": title,
                "intro": intro,
                "lm_description": lm_desc,
                "to_index_text": to_index_text,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written: {written} -> {OUTPUT_PATH}")
    print(f"Cache dir: {CACHE_DIR}")


if __name__ == "__main__":
    main()
