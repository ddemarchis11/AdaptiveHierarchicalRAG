import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "subset_tables_only.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "intros_to_index.jsonl"


def safe_load_json(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    written = 0
    seen = set()

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            table = safe_load_json(line)
            if not table:
                continue

            table_id = table.get("table_id")
            if not table_id or table_id in seen:
                continue

            title = table.get("title", "") or ""
            intro = table.get("intro", "") or ""
            url = table.get("url", "") or ""
            section_title = table.get("section_title", "") or ""

            intro_norm = norm_ws(intro)
            if not intro_norm:
                continue

            out = {
                "doc_id": f"{table_id}__intro",
                "table_id": table_id,
                "title": title,
                "intro": intro,
                "section_title": section_title,
                "url": url,
                "serialization": "intro_only",
                "to_index_text": intro_norm,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            seen.add(table_id)
            written += 1

    print(f"Written: {written} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
