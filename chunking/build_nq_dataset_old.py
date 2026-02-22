from datasets import load_dataset
import os, json, random, re

DOCS_TO_BUILD = 1000
SPLIT = "test"             
STITCH_GROUP_SIZE = 5      
SEP = "\n\n"               
SEED = 42

OUT_DIR = "nq_docs"
os.makedirs(OUT_DIR, exist_ok=True)

def main():

    ds = load_dataset("LLukas22/nq-simplified", split=SPLIT)

    indices = list(range(len(ds)))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    docs = []
    ptr = 0

    for doc_i in range(DOCS_TO_BUILD):
        group = indices[ptr:ptr + STITCH_GROUP_SIZE]
        if len(group) < STITCH_GROUP_SIZE:
            break
        ptr += STITCH_GROUP_SIZE

        stitched_parts = []
        qas = []
        offset = 0

        for idx in group:
            row = ds[idx]
            ctx = row["context"]
            stitched_parts.append(ctx)

            ans_texts = row["answers"]["text"]
            ans_starts = row["answers"]["answer_start"]

            answers = []
            for text, start in zip(ans_texts, ans_starts):
                if text is None or text == "":
                    continue
                answers.append({
                    "text": text,
                    "answer_start": int(start) + offset
                })

            if answers:
                qas.append({
                    "question": row["question"],
                    "id": f"nq_{SPLIT}_{idx}",
                    "answers": answers,
                })

            offset += len(ctx) + len(SEP)

        if not qas:
            continue

        doc = {
            "title": f"nq_stitched_{doc_i:06d}",
            "context": SEP.join(stitched_parts),
            "qas": qas,
        }
        docs.append(doc)
        
        os.makedirs("nq_docs", exist_ok=True)

    for i, doc in enumerate(docs):
        safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", doc["title"])
        filename = f"{OUT_DIR}/{i:04d}_{safe_title}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"Documenti stitched salvati: {len(docs)} in {OUT_DIR}/")
    print(f"Query totali (circa): {sum(len(d['qas']) for d in docs)}")

if __name__ == "__main__": main()