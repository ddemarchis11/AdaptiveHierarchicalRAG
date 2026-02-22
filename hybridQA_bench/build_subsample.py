#!/usr/bin/env python3
import json
import random
import os

INPUT_QAS = "qas.jsonl"
INPUT_TABLES = "tables.jsonl"

OUTPUT_TABLES = "subset_tables_only.jsonl"
OUTPUT_QAS = "subset_qas_only.jsonl"

SAMPLE_SIZE = 400
RANDOM_SEED = 42

def main():
    if not os.path.exists(INPUT_QAS) or not os.path.exists(INPUT_TABLES):
        return

    valid_table_ids = set()
    with open(INPUT_QAS, "r", encoding="utf-8") as f:
        for line in f:
            try:
                qa = json.loads(line)
            except json.JSONDecodeError:
                continue
            if qa.get("table_only") is True and "table_id" in qa:
                valid_table_ids.add(qa["table_id"])

    valid_table_ids_list = sorted(valid_table_ids)

    if SAMPLE_SIZE and len(valid_table_ids_list) > SAMPLE_SIZE:
        random.seed(RANDOM_SEED)
        selected_ids = set(random.sample(valid_table_ids_list, SAMPLE_SIZE))
    else:
        selected_ids = set(valid_table_ids_list)

    saved_tables = 0
    with open(INPUT_TABLES, "r", encoding="utf-8") as fin, open(OUTPUT_TABLES, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                table = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = table.get("table_id")
            if tid in selected_ids:
                fout.write(line)
                saved_tables += 1

    saved_qas = 0
    with open(INPUT_QAS, "r", encoding="utf-8") as fin, open(OUTPUT_QAS, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                qa = json.loads(line)
            except json.JSONDecodeError:
                continue
            if qa.get("table_only") is True and qa.get("table_id") in selected_ids:
                fout.write(line)
                saved_qas += 1

    print(f"Selected table_ids: {len(selected_ids)}")
    print(f"Saved tables: {saved_tables} -> {OUTPUT_TABLES}")
    print(f"Saved qas: {saved_qas} -> {OUTPUT_QAS}")

if __name__ == "__main__":
    main()
