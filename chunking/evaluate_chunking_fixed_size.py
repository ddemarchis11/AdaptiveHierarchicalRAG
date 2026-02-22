import numpy as np
import glob
import json
import statistics
import os
from tqdm import tqdm
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from chunking.strategies.semantic_chunking import semantic_chunking
from chunking.strategies.fixed_chunking import fixed_token_chunking

MODEL_NAME = "baai/bge-large-en-v1.5"
RANKS = [1, 2, 3, 4, 5]
DOCS_PATH = "datasets/nq_docs/*.json"
FIXED_WINDOW_SIZE = 256

model = SentenceTransformer(MODEL_NAME)

def build_qa_text_list(filepath: str) -> Tuple[List[Dict], str]:
    with open(filepath, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["qas"], doc["context"]

def run_evaluation(data_list: List[Tuple[List, List]], label: str) -> Dict:
    hits = {k: 0 for k in RANKS}
    total_queries = 0
    token_lengths = []

    for chunks, qa_list in tqdm(data_list, desc=label):
        if not chunks: continue
        
        chunk_texts = [c.text if hasattr(c, 'text') else c for c in chunks]
        chunk_lens = [c.length if hasattr(c, 'length') else len(model.tokenizer.encode(c)) for c in chunks]
        
        embeddings = model.encode(chunk_texts, normalize_embeddings=True, convert_to_numpy=True)
        
        for qa in qa_list:
            if not qa["answers"]: continue
            
            query_emb = model.encode(qa["question"], normalize_embeddings=True, convert_to_numpy=True)
            scores = embeddings @ query_emb
            top_indices = np.argsort(-scores)[:max(RANKS)]
            
            total_queries += 1
            ans_lower = qa["answers"][0]["text"].lower()
            
            for k in RANKS:
                indices_to_check = top_indices[:k]
                if any(ans_lower in chunk_texts[idx].lower() for idx in indices_to_check):
                    hits[k] += 1
            
            token_lengths.append(chunk_lens[top_indices[0]])
    
    avg_len = statistics.mean(token_lengths) if token_lengths else 0

    results = {}
    for k in RANKS:
        results[f'Hit@{k}'] = hits[k] / total_queries if total_queries > 0 else 0
        
    results['total'] = total_queries
    results['avg_len'] = avg_len
    return results

files = glob.glob(DOCS_PATH)
if not files: exit()

semantic_data = []
fixed_data = []

print(f"Processando {len(files)} documenti...")

for f in files:
    qas, text = build_qa_text_list(f)
    
    result_s = semantic_chunking(text, model)
    if isinstance(result_s, tuple) and len(result_s) == 2:
        s_chunks, s_lens = result_s
    else:
        s_chunks = result_s
    semantic_data.append((s_chunks, qas))
    
    f_chunks = fixed_token_chunking(text, window_size=FIXED_WINDOW_SIZE)
    fixed_data.append((f_chunks, qas))

res_semantic = run_evaluation(semantic_data, "Semantic")
res_fixed = run_evaluation(fixed_data, f"Fixed-{FIXED_WINDOW_SIZE}")

markdown_content = f"""# Benchmark Report: Semantic vs Fixed

| Parametro | Valore |
| :--- | :--- |
| **Modello Embedding** | `{MODEL_NAME}` |
| **Fixed Window Size** | {FIXED_WINDOW_SIZE} |
| **Documenti Testati** | {len(files)} |
| **Query Totali** | {res_semantic['total']} |

## Analisi Comparativa

| Metrica | Semantic Chunking | Fixed Token Chunking |
| :--- | :--- | :--- |
| **Avg Chunk Length (Top-1)** | {res_semantic['avg_len']:.2f} | {res_fixed['avg_len']:.2f} |
"""

for k in RANKS:
    h_s = res_semantic[f'Hit@{k}']
    h_f = res_fixed[f'Hit@{k}']
    markdown_content += f"| **Hit@{k}** | {h_s:.4f} | {h_f:.4f} |\n"

os.makedirs("benchmarks", exist_ok=True)
output_filename = "benchmarks/benchmark_results_final.md"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"Risultati salvati in {output_filename}")