import json
import sys
import time
import logging
from statistics import mean
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from config import EvaluationConfig
    from llm_stub import get_eval_client
except ImportError:
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_files(directory: Path) -> List[Dict[str, Any]]:
    if not directory.exists():
        logger.error(f"Directory non trovata: {directory}")
        return []
        
    data = []
    files = list(directory.glob("*.json"))
    files.sort()
    
    print(f"Trovati {len(files)} file di risultati in {directory}")
    
    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, dict):
                    if "id" not in content:
                        content["id"] = f_path.stem
                    data.append(content)
                elif isinstance(content, list):
                    for item in content:
                        data.append(item)
        except Exception as e:
            logger.warning(f"Errore lettura file {f_path}: {e}")
            
    return data

def parse_context(context_data: Any) -> str:
    if isinstance(context_data, str):
        return context_data
    if isinstance(context_data, list):
        return "\n\n---\n\n".join([
            str(c.get("text", "")).strip() 
            for c in context_data 
            if isinstance(c, dict) and "text" in c
        ])
    return ""

def call_judge_llm(user_prompt: str, cfg: EvaluationConfig, client) -> Dict[str, Any]:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        except Exception:
            time.sleep(1)
            
    return {
        "answer_correctness": 0.0,
        "faithfulness_to_context": 0.0,
        "evidence_coverage": 0.0,
        "explanation": "API Error or Invalid JSON after retries"
    }

def evaluate_run():
    cfg = EvaluationConfig()
    client = get_eval_client()
    
    input_dir = current_dir / "hotpotQA_bench_eval/graph_rag_ambiguous" / "llm_results_llama"
    eval_dir = current_dir / "hotpotQA_bench_eval/graph_rag_ambiguous" / "llm_evaluations_llama"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    examples = load_json_files(input_dir)
    if not examples:
        return

    metrics = {
        "correctness": [],
        "faithfulness": [],
        "coverage": []
    }
    
    print(f"Inizio valutazione di {len(examples)} risposte con {cfg.model}...")

    for ex in tqdm(examples):
        ex_id = ex.get("id", "unknown")
        safe_name = str(ex_id).replace("/", "_") + ".json"
        out_path = eval_dir / safe_name
        
        if out_path.exists():
            with open(out_path, 'r') as f:
                saved_res = json.load(f)
                s = saved_res.get("scores", {})
                metrics["correctness"].append(float(s.get("answer_correctness", 0)))
                metrics["faithfulness"].append(float(s.get("faithfulness_to_context", 0)))
                metrics["coverage"].append(float(s.get("evidence_coverage", 0)))
            continue

        question = ex.get("question", "")
        gold_answer = ex.get("gold_answer", "")
        llm_answer = ex.get("llm_answer", "")
        context_txt = parse_context(ex.get("context", []))

        user_prompt = f"""
        QUESTION: {question}
        
        GOLD ANSWER (GROUND TRUTH): 
        {gold_answer}
        
        RETRIEVED CONTEXT:
        {context_txt[:15000]} 
        
        MODEL ANSWER:
        {llm_answer}
        """
        
        scores = call_judge_llm(user_prompt, cfg, client)
        
        result_record = {
            "id": ex_id,
            "question": question,
            "gold_answer": gold_answer,
            "llm_answer": llm_answer,
            "scores": scores
        }
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_record, f, ensure_ascii=False, indent=2)
            
        metrics["correctness"].append(float(scores.get("answer_correctness", 0)))
        metrics["faithfulness"].append(float(scores.get("faithfulness_to_context", 0)))
        metrics["coverage"].append(float(scores.get("evidence_coverage", 0)))

    if metrics["correctness"]:
        print("\n" + "="*40)
        print("RISULTATI VALUTAZIONE")
        print("="*40)
        print(f"Avg Answer Correctness:    {mean(metrics['correctness']):.3f}")
        print(f"Avg Faithfulness:          {mean(metrics['faithfulness']):.3f}")
        print(f"Avg Evidence Coverage:     {mean(metrics['coverage']):.3f}")
        print(f"Total Evaluated:           {len(metrics['correctness'])}")
        
        stats = {
            "avg_correctness": mean(metrics['correctness']),
            "avg_faithfulness": mean(metrics['faithfulness']),
            "avg_coverage": mean(metrics['coverage']),
            "total_evaluated": len(metrics['correctness'])
        }
        with open(eval_dir / "summary_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    evaluate_run()