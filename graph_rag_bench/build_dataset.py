#!/usr/bin/env python3
import json
import pathlib
import requests

NOVEL_CORPUS_URL = (
    "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench"
    "/resolve/main/Datasets/Corpus/novel.json"
)
NOVEL_QUESTIONS_URL = (
    "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench"
    "/resolve/main/Datasets/Questions/novel_questions.json"
)

def download_json(url: str):
    print(f"Downloading {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    return json.loads(resp.text)


def write_jsonl(path: pathlib.Path, records):
    print(f"Writing {path}")
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def main():
    out_dir = pathlib.Path(".") 
    contents_path = out_dir / "contents.jsonl"
    questions_path = out_dir / "questions.jsonl"

    corpus = download_json(NOVEL_CORPUS_URL)

    write_jsonl(contents_path, corpus)

    questions_raw = download_json(NOVEL_QUESTIONS_URL)

    questions_proc = []
    for q in questions_raw:
        rec = {
            "id": q.get("id"),
            "source": q.get("source"), 
            "answer": q.get("answer"),
            "question_type": q.get("question_type"),
            "evidence": q.get("evidence", []),
        }
        if "evidence_triple" in q:
            rec["evidence_triple"] = q["evidence_triple"]

        questions_proc.append(rec)

    write_jsonl(questions_path, questions_proc)

    print("\Done!")
    print(f"- Novel Corpus:   {contents_path.resolve()}")
    print(f"- Novel Questions:  {questions_path.resolve()}")


if __name__ == "__main__":
    main()