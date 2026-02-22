import json
import random
import math
from pathlib import Path

def calculate_sample_size(N, confidence_level=1.96, margin_error=0.05, p=0.5):
    numerator = (N * (confidence_level**2) * p * (1-p))
    denominator = ((margin_error**2) * (N - 1)) + ((confidence_level**2) * p * (1-p))
    return math.ceil(numerator / denominator)

def process_representative_sample(input_path, output_path):
    if not Path(input_path).exists():
        print(f"Errore: Il file {input_path} non esiste.")
        return

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    N = len(records)
    target_n = calculate_sample_size(N)
    
    by_type = {}
    for r in records:
        q_type = r.get('question_type', 'Unknown')
        if q_type not in by_type:
            by_type[q_type] = []
        by_type[q_type].append(r)

    print(f"{'='*45}")
    print(f"ANALISI DATASET (N={N})")
    print(f"Target rappresentativo (95%/5%): {target_n} record")
    print(f"{'='*45}")

    subset = []
    for q_type, type_records in by_type.items():
        proportion = len(type_records) / N
        n_to_extract = math.floor(target_n * proportion)
        
        n_to_extract = max(1, n_to_extract)
        
        random.shuffle(type_records)
        selected = type_records[:n_to_extract]
        subset.extend(selected)
        
        print(f"- {q_type:22}: {len(type_records):4} rec -> Estratti {len(selected)}")

    if len(subset) < target_n:
        remaining = [r for r in records if r not in subset]
        subset.extend(random.sample(remaining, target_n - len(subset)))
    
    random.shuffle(subset)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in subset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n{'='*45}")
    print(f"COMPLETATO: {len(subset)} record salvati in {output_path}")
    print(f"{'='*45}")

if __name__ == "__main__":
    INPUT_FILE = "questions.jsonl"
    OUTPUT_FILE = "questions_subsample.jsonl"
    
    process_representative_sample(INPUT_FILE, OUTPUT_FILE)