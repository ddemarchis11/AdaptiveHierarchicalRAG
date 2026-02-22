import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "subset_tables_only.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "rows_to_index.jsonl"


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


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")

    written = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            table = safe_load_json(line)
            if not table:
                continue

            table_id = table.get("table_id")
            if not table_id:
                continue

            title = table.get("title", "") or ""
            intro = table.get("intro", "") or ""
            section_title = table.get("section_title", "") or ""
            url = table.get("url", "") or ""

            header = [normalize_ws(str(h)) for h in (table.get("header", []) or [])]
            rows = table.get("rows", []) or []

            title_norm = normalize_ws(title)
            intro_norm = normalize_ws(intro)

            for r_idx, row in enumerate(rows):
                if not isinstance(row, list):
                    continue

                row_text = row_to_kv(header, row)
                if not row_text:
                    continue

                full_text_parts = []
                if title_norm:
                    full_text_parts.append(title_norm)
                if intro_norm:
                    full_text_parts.append(intro_norm)
                full_text_parts.append(row_text)

                to_index_text = "\n".join(full_text_parts).strip()

                out_obj = {
                    "doc_id": f"{table_id}__row_{r_idx}",
                    "table_id": table_id,
                    "row_index": r_idx,
                    "title": title,
                    "intro": intro,
                    "section_title": section_title,
                    "url": url,
                    "row_text": row_text,
                    "to_index_text": to_index_text,
                    "meta": {
                        "n_cols": len(header),
                    },
                }

                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                written += 1

    print(f"Written: {written} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
