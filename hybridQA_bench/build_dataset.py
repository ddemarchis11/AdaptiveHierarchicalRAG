#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

TABLES_DIR = os.path.join("data", "tables_out")
QA_DIR = os.path.join("data", "released_data") 

OUT_QAS = "qas.jsonl"
OUT_TABLES = "tables.jsonl"

_ws_re = re.compile(r"\s+")

def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = _ws_re.sub(" ", s)
    return s.lower()

def safe_jsonl_write(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

def linearize_table(header: List[str], rows: List[List[str]]) -> str:
    lines = []
    for idx, row in enumerate(rows):
        parts = []
        for c_idx, cell_val in enumerate(row):
            col_name = header[c_idx] if c_idx < len(header) else f"col_{c_idx}"
            parts.append(f"{col_name} is {cell_val}")
        lines.append(f"row {idx+1}: " + "; ".join(parts))
    return " | ".join(lines)

def extract_passages_text(cell_passages: List[List[List[Dict]]]) -> str:
    texts = []
    for r_idx, row_p in enumerate(cell_passages):
        for c_idx, cell_p_list in enumerate(row_p):
            for item in cell_p_list:
                if "summary" in item:
                    texts.append(item["summary"])
    return " ".join(texts)

def load_tables_hybridqa(target_dir: str) -> Dict[str, Any]:
    tables = {}
    files = glob.glob(os.path.join(target_dir, "*.json"))
    
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            
            tid = obj.get("table_id")
            if not tid:
                tid = os.path.splitext(os.path.basename(path))[0]
            
            tables[tid] = obj
        except Exception:
            continue
            
    return tables

def find_split_file(base_dir: str, split: str) -> str:
    direct = os.path.join(base_dir, f"{split}.json")
    if os.path.isfile(direct):
        return direct
    
    pats = [
        os.path.join(base_dir, f"{split}.json"),
        os.path.join(base_dir, "**", f"{split}.json"),
        os.path.join(base_dir, f"*{split}*.json")
    ]
    for p in pats:
        matches = glob.glob(p, recursive=True)
        matches = [m for m in matches if "table" not in os.path.basename(m).lower()]
        if matches:
            matches.sort(key=len)
            return matches[0]
            
    raise FileNotFoundError(f"Non trovo il file QA per split={split} in {base_dir}")

def main():
    if not os.path.isdir(TABLES_DIR):
        raise FileNotFoundError(f"Directory tabelle non trovata: {TABLES_DIR}")

    raw_tables = load_tables_hybridqa(TABLES_DIR)
    
    with open(OUT_TABLES, "w", encoding="utf-8") as ft:
        for tid, data in raw_tables.items():
            header = data.get("header", [])
            rows = data.get("rows", [])
            title = data.get("title", "")
            intro = data.get("intro", "")
            sect_title = data.get("section_title", "")
            passages_struct = data.get("cell_passages", [])

            table_str = linearize_table(header, rows)
            passages_text = extract_passages_text(passages_struct)

            full_context = (
                f"Title: {title}\n"
                f"Intro: {intro}\n"
                f"Section: {sect_title}\n"
                f"Table Data: {table_str}\n"
                f"Related Passages: {passages_text}"
            )

            out_obj = {
                "table_id": tid,
                "url": data.get("url", ""),
                "title": title,
                "intro": intro,
                "header": header,
                "rows": rows,
                "cell_passages": passages_struct,
                "llm_context": full_context, 
                "table_linearized": table_str,
                "passages_text": passages_text,
                "meta": {
                    "n_rows": len(rows),
                    "n_cols": len(header)
                }
            }
            safe_jsonl_write(ft, out_obj)

    splits = ["train", "dev", "test"]
    
    with open(OUT_QAS, "w", encoding="utf-8") as fq:
        for split in splits:
            try:
                split_path = find_split_file(QA_DIR, split)
            except FileNotFoundError:
                continue

            with open(split_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
                
            if isinstance(qa_data, dict) and "data" in qa_data:
                qa_items = qa_data["data"]
            else:
                qa_items = qa_data

            for item in qa_items:
                qid = item.get("question_id") or item.get("id")
                question = item.get("question")
                tid = item.get("table_id")
                
                answer_text = item.get("answer-text")
                if answer_text is None:
                    answer_text = item.get("answer_text") or item.get("answer")

                if not qid or not tid:
                    continue

                is_table_only = False
                if tid in raw_tables and answer_text:
                    t_data = raw_tables[tid]
                    table_values = [str(x) for x in t_data.get("header", [])]
                    for r in t_data.get("rows", []):
                        table_values.extend([str(c) for c in r])
                    
                    norm_ans = norm_text(answer_text)
                    for val in table_values:
                        nv = norm_text(val)
                        if norm_ans == nv:
                            is_table_only = True
                            break
                        if len(norm_ans) > 2 and norm_ans in nv and len(nv) < 100:
                            is_table_only = True
                            break

                out_obj = {
                    "question_id": qid,
                    "split": split,
                    "table_id": tid,
                    "question": question,
                    "answer": answer_text,
                    "table_only": is_table_only,
                }
                safe_jsonl_write(fq, out_obj)

if __name__ == "__main__":
    main()