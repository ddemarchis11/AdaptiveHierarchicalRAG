from datasets import load_dataset
from collections import defaultdict
import os, re, json

DOCS_TO_BUILD = 1000

def main():

    squad = load_dataset("squad")
    train = squad["train"]

    context2data = defaultdict(list)
    for row in train:
        context2data[row["context"]].append(row)

    docs = []

    for context, rows in list(context2data.items())[:DOCS_TO_BUILD]:
        qas = []
        
        for row in rows:
            answers = []
            for text, start in zip(row["answers"]["text"], row["answers"]["answer_start"]):
                answers.append({
                    "text": text,
                    "answer_start": start,
                })

            qas.append({
                "question": row["question"],
                "id": row["id"],
                "answers": answers,
            })

        docs.append({
            "title": rows[0]["title"],
            "context": context,
            "qas": qas,
        })

    print(f"Documenti creati: {len(docs)}")

    os.makedirs("squad_docs", exist_ok=True)

    for i, doc in enumerate(docs):
        safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", doc["title"])
        filename = f"squad_docs/{i:02d}_{safe_title}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        print("Salvato:", filename)
    
if __name__ == "__main__": main()