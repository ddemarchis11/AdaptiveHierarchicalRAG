import sys
import json
import argparse
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

print(f"\n{'='*60}")
print(f"PROJECT ROOT: {project_root}")
print(f"SCRIPT DIR:   {current_dir}")

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from config import DEFAULT_CONFIG, PipelineConfig
    from llm_stub import get_llm_client

    import cot_rag
    import graph_rag

    from neo4j import GraphDatabase
    from elasticsearch import Elasticsearch
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"Errore Importazione moduli: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.ERROR)
logging.getLogger("cot_rag_second").setLevel(logging.INFO)

def load_jsonl(filename: str) -> List[Dict[str, Any]]:
    data = []
    file_path = project_root / "hotpotQA" / filename

    print(f"Cercando il file di input in: {file_path}")

    if not file_path.exists():
        print(f"ERRORE: File input non trovato in: {file_path}")
        return []

    print(f"File input trovato.")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def get_pipeline_runner(mode: str, cfg: PipelineConfig, llm_client):

    if mode == "cot_rag":
        print("Inizializzazione Retriever (ElasticSearch)...")
        es_retriever = cot_rag.build_es_retriever_from_cfg(cfg)

        print("Configurazione: CoT RAG")
        cot_cfg = cot_rag.CoTRAGConfig(
            k_pool_per_step=20,
            top_k_step_context=5,
            max_steps=3,
            verbose=False
        )

        backend = cot_rag.ClassicHybridBackend(es_retriever, cot_cfg)
        chain = cot_rag.build_cot_rag_chain(backend, llm_client, cot_cfg)

        def run_cot(question):
            res = chain.invoke({"question": question})
            context_list = []
            if "documents" in res:
                for d in res["documents"]:
                    context_list.append({"text": d.get("text", ""), "id": d.get("id")})

            return {
                "answer": res["answer"],
                "context_list": context_list,
                "scratchpad": res.get("scratchpad", ""),
                "history": res.get("history", [])
            }

        return run_cot

    if mode == "classic":
        print("Inizializzazione Retriever (ElasticSearch)...")
        es_retriever = cot_rag.build_es_retriever_from_cfg(cfg)

        print("Configurazione: Classic RAG (No CoT)")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question based only on the following context:\n\n{context}"),
            ("user", "{question}")
        ])
        chain = prompt | llm_client | StrOutputParser()

        def run_classic(question):
            docs = es_retriever.retrieve(question, k=5)
            context_text = "\n\n".join([d["text"] for d in docs])
            context_list = [{"text": d["text"], "id": d["id"]} for d in docs]
            answer = chain.invoke({"question": question, "context": context_text})
            return {
                "answer": answer,
                "context_list": context_list,
            }

        return run_classic

    if mode == "graph":
        print("Configurazione: Graph RAG")
        print("Inizializzazione client ES e Neo4j (condivisi tra le query)...")

        es = graph_rag.get_es_client(
            url=cfg.es.url,
            user=getattr(cfg.es, "user", None),
            password=getattr(cfg.es, "password", None),
            verify_certs=getattr(cfg.es, "verify_certs", True),
        )
        driver = GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        print("Client ES e Neo4j pronti.")

        def run_graph(question):
            res = graph_rag.answer_query(
                query=question,
                config=cfg,
                es_client=es,
                neo4j_driver=driver,
            )

            chunks = res.get("chunks", [])
            context_list = [
                {
                    "text": c.get("text", ""),
                    "id": c.get("chunk_id"),
                    "doc_id": c.get("doc_id"),
                    "community_id": c.get("community_id"),
                    "score": c.get("score"),
                }
                for c in chunks
            ]

            answer = res.get("answer", "")
            if hasattr(answer, "content"):
                answer = answer.content

            return {
                "answer": answer,
                "context_list": context_list,
                "communities": res.get("communities", []),
                "debug": res.get("debug", {}),
            }

        return run_graph

    raise ValueError(f"Modalità '{mode}' non supportata (usa 'cot_rag', 'classic' o 'graph')")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cot_rag", choices=["cot_rag", "classic", "graph"])
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG

    if cfg.es.password:
        print("Configurazione caricata correttamente (Password ES trovata).")
    else:
        print("ERRORE CRITICO: Password ElasticSearch non trovata in cfg.es.password.")
        print("Controlla che il file .env sia corretto e raggiungibile da config.py")
        return

    print(f"{'='*60}\n")

    try:
        llm_client = get_llm_client(cfg)
        runner = get_pipeline_runner(args.mode, cfg, llm_client)
    except Exception as e:
        print(f"Errore inizializzazione Pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    mode_dir_map = {
        "cot_rag": "cot_rag_evaluation",
        "classic": "classic_rag_evaluation",
        "graph":   "graph_rag_evaluation",
    }
    out_dir = current_dir / "hotpotQA_bench_eval" / mode_dir_map[args.mode] / "llm_results_kimi"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCARTELLA OUTPUT RISULTATI: {out_dir.resolve()}")

    examples = load_jsonl(cfg.evaluation.input_file)
    if not examples:
        return

    print(f"Inizio elaborazione di {len(examples)} domande...")

    for i, ex in enumerate(tqdm(examples)):
        question = ex["question"]
        ex_id = ex.get("id", f"{args.mode}_{i}")

        safe_name = str(ex_id).replace("/", "_") + ".json"
        out_path = out_dir / safe_name

        if out_path.exists():
            continue

        try:
            res = runner(question)

            record = {
                "id": ex_id,
                "question": question,
                "gold_answer": ex.get("answer"),
                "llm_answer": res["answer"],
                "context": res["context_list"],
            }

            if args.mode == "cot_rag":
                record["cot_scratchpad"] = res.get("scratchpad")
                record["cot_history"] = res.get("history")
            elif args.mode == "graph":
                record["communities"] = res.get("communities")
                record["debug"] = res.get("debug")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Errore ID {ex_id}: {e}")
            continue

    print(f"\nFinito! Risultati salvati in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()