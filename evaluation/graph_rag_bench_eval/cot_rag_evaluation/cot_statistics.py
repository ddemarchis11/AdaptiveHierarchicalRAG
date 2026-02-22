from __future__ import annotations

import json
import math
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

EVAL_DIRS = {
    "kimi": Path("llm_evaluations_kimi"),
    "llama": Path("llm_evaluations_llama"),
}

RESULT_DIRS = {
    "kimi": Path("llm_results_kimi"),
    "llama": Path("llm_results_llama"),
}

GRAPH_RAG_BENCH_JSONL = Path("../../graph_rag_bench/questions_subsample.jsonl")
OUT_MD = Path("comparison.md")

METRICS = ["answer_correctness", "faithfulness_to_context", "evidence_coverage"]

def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")

def safe_stdev(xs: List[float]) -> float:
    return float(stats.pstdev(xs)) if len(xs) >= 2 else 0.0

def fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"

def iter_json_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.json") if p.is_file() and p.name != "summary_stats.json"])

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_question_types_by_id(jsonl_path: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if not jsonl_path.exists():
        return m

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                _id = str(obj.get("id", ""))
                qt = str(obj.get("question_type", "UNKNOWN"))
                if _id:
                    m[_id] = qt
            except Exception:
                continue
    return m

def load_evaluations(folder: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for fp in iter_json_files(folder):
        try:
            obj = load_json(fp)
            _id = str(obj.get("id", fp.stem))
            out[_id] = {
                "id": _id,
                "question": str(obj.get("question", "")),
                "scores": obj.get("scores", {}) or {},
                "filepath": str(fp),
            }
        except Exception:
            continue
    return out

def load_results(folder: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for fp in iter_json_files(folder):
        try:
            obj = load_json(fp)
            _id = str(obj.get("id", fp.stem))
            out[_id] = {
                "id": _id,
                "question": str(obj.get("question", "")),
                "cot_history": obj.get("cot_history", []) or [],
                "filepath": str(fp),
            }
        except Exception:
            continue
    return out

def cot_steps_from_history(hist: Any) -> Optional[int]:
    if isinstance(hist, list):
        return len(hist)
    return None

def build_index_by_id(records: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return records

def paired_by_id(
    eval_a: Dict[str, Dict[str, Any]],
    eval_b: Dict[str, Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    ia = eval_a
    ib = eval_b
    common_ids = sorted(set(ia.keys()) & set(ib.keys()))
    return [(ia[_id], ib[_id], _id) for _id in common_ids]

def collect_metric_values(records: List[Dict[str, Any]], metric: str) -> List[float]:
    xs: List[float] = []
    for r in records:
        v = r.get("scores", {}).get(metric, None)
        if isinstance(v, (int, float)):
            xs.append(float(v))
    return xs

def aggregate_by_question_type(
    paired: List[Tuple[Dict[str, Any], Dict[str, Any], str]],
    qtype_map: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any], str]]] = {}
    for a, b, _id in paired:
        qt = qtype_map.get(_id, "UNKNOWN")
        buckets.setdefault(qt, []).append((a, b, _id))

    out: Dict[str, Dict[str, Any]] = {}
    for qt, items in buckets.items():
        a_items = [x[0] for x in items]
        b_items = [x[1] for x in items]
        payload: Dict[str, Any] = {"n": len(items), "metrics": {}}
        for m in METRICS:
            xa = collect_metric_values(a_items, m)
            xb = collect_metric_values(b_items, m)
            deltas: List[float] = []
            for (ra, rb, _) in items:
                va = ra.get("scores", {}).get(m, None)
                vb = rb.get("scores", {}).get(m, None)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    deltas.append(float(vb) - float(va))
            payload["metrics"][m] = {
                "a_mean": safe_mean(xa),
                "b_mean": safe_mean(xb),
                "delta_mean_b_minus_a": safe_mean(deltas),
            }
        out[qt] = payload
    return out

def global_summary(
    paired: List[Tuple[Dict[str, Any], Dict[str, Any], str]]
) -> Dict[str, Any]:
    a_items = [x[0] for x in paired]
    b_items = [x[1] for x in paired]
    res: Dict[str, Any] = {
        "n_paired": len(paired),
        "metrics": {},
    }
    for m in METRICS:
        xa = collect_metric_values(a_items, m)
        xb = collect_metric_values(b_items, m)
        deltas: List[float] = []
        for (ra, rb, _) in paired:
            va = ra.get("scores", {}).get(m, None)
            vb = rb.get("scores", {}).get(m, None)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                deltas.append(float(vb) - float(va))
        res["metrics"][m] = {
            "a_mean": safe_mean(xa),
            "a_stdev": safe_stdev(xa),
            "b_mean": safe_mean(xb),
            "b_stdev": safe_stdev(xb),
            "delta_mean_b_minus_a": safe_mean(deltas),
            "delta_stdev": safe_stdev(deltas),
        }
    return res

def steps_summary(
    res_kimi: Dict[str, Dict[str, Any]],
    res_llama: Dict[str, Dict[str, Any]],
    qtype_map: Dict[str, str],
) -> Dict[str, Any]:
    paired = paired_by_id(res_kimi, res_llama)
    overall_steps = {"kimi": [], "llama": [], "delta_llama_minus_kimi": []}
    by_type: Dict[str, Dict[str, List[int]]] = {}

    for a, b, _id in paired:
        sa = cot_steps_from_history(a.get("cot_history"))
        sb = cot_steps_from_history(b.get("cot_history"))
        if sa is None or sb is None:
            continue
        overall_steps["kimi"].append(sa)
        overall_steps["llama"].append(sb)
        overall_steps["delta_llama_minus_kimi"].append(sb - sa)

        qt = qtype_map.get(_id, "UNKNOWN")
        by_type.setdefault(qt, {"kimi": [], "llama": [], "delta": []})
        by_type[qt]["kimi"].append(sa)
        by_type[qt]["llama"].append(sb)
        by_type[qt]["delta"].append(sb - sa)

    out: Dict[str, Any] = {
        "n_paired": len(paired),
        "overall": {
            "kimi_mean_steps": safe_mean([float(x) for x in overall_steps["kimi"]]),
            "llama_mean_steps": safe_mean([float(x) for x in overall_steps["llama"]]),
            "delta_mean_llama_minus_kimi": safe_mean([float(x) for x in overall_steps["delta_llama_minus_kimi"]]),
        },
        "by_question_type": {},
    }

    for qt, d in sorted(by_type.items(), key=lambda x: x[0]):
        out["by_question_type"][qt] = {
            "n": len(d["kimi"]),
            "kimi_mean_steps": safe_mean([float(x) for x in d["kimi"]]),
            "llama_mean_steps": safe_mean([float(x) for x in d["llama"]]),
            "delta_mean_llama_minus_kimi": safe_mean([float(x) for x in d["delta"]]),
        }
    return out

def render_md(
    eval_kimi: Dict[str, Dict[str, Any]],
    eval_llama: Dict[str, Dict[str, Any]],
    res_kimi: Dict[str, Dict[str, Any]],
    res_llama: Dict[str, Dict[str, Any]],
    qtype_map: Dict[str, str],
) -> str:
    paired_eval = paired_by_id(eval_kimi, eval_llama)
    summary = global_summary(paired_eval)
    by_type = aggregate_by_question_type(paired_eval, qtype_map)
    step_stats = steps_summary(res_kimi, res_llama, qtype_map)

    all_eval_ids = set(eval_kimi.keys()) | set(eval_llama.keys())
    n_total_ids = len(all_eval_ids)
    n_mapped = sum(1 for _id in all_eval_ids if _id in qtype_map)

    lines: List[str] = []
    lines.append("# LLM Evaluation Comparison (Kimi vs Llama)\n")
    lines.append(f"- Paired (by ID) evaluation items: **{summary['n_paired']}**")
    lines.append(f"- GraphRAG-Bench question type mapping coverage: **{n_mapped}/{n_total_ids}** mapped\n")

    lines.append("## Overall judging score statistics\n")
    lines.append("| metric | kimi mean | kimi stdev | llama mean | llama stdev | Δ (llama - kimi) mean | Δ stdev |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in METRICS:
        d = summary["metrics"][m]
        lines.append(
            f"| `{m}` | {fmt(d['a_mean'])} | {fmt(d['a_stdev'])} | {fmt(d['b_mean'])} | {fmt(d['b_stdev'])} | {fmt(d['delta_mean_b_minus_a'])} | {fmt(d['delta_stdev'])} |"
        )
    lines.append("")

    lines.append("## Judging scores by question type\n")
    lines.append("| question_type | n | metric | kimi mean | llama mean | Δ (llama - kimi) mean |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for qt, payload in sorted(by_type.items(), key=lambda x: x[0]):
        n = payload["n"]
        for m in METRICS:
            mm = payload["metrics"][m]
            lines.append(
                f"| `{qt}` | {n} | `{m}` | {fmt(mm['a_mean'])} | {fmt(mm['b_mean'])} | {fmt(mm['delta_mean_b_minus_a'])} |"
            )
    lines.append("")

    lines.append("## CoT step statistics\n")
    lines.append(f"- Paired (by ID) results items: **{step_stats['n_paired']}**\n")
    lines.append("| scope | kimi mean steps | llama mean steps | Δ (llama - kimi) mean |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| overall | {fmt(step_stats['overall']['kimi_mean_steps'])} | {fmt(step_stats['overall']['llama_mean_steps'])} | {fmt(step_stats['overall']['delta_mean_llama_minus_kimi'])} |"
    )
    lines.append("")
    if step_stats["by_question_type"]:
        lines.append("### Steps by question type\n")
        lines.append("| question_type | n | kimi mean steps | llama mean steps | Δ (llama - kimi) mean |")
        lines.append("|---|---:|---:|---:|---:|")
        for qt, d in sorted(step_stats["by_question_type"].items(), key=lambda x: x[0]):
            lines.append(
                f"| `{qt}` | {d['n']} | {fmt(d['kimi_mean_steps'])} | {fmt(d['llama_mean_steps'])} | {fmt(d['delta_mean_llama_minus_kimi'])} |"
            )
        lines.append("")

    return "\n".join(lines)

def main() -> int:
    eval_kimi = load_evaluations(EVAL_DIRS["kimi"])
    eval_llama = load_evaluations(EVAL_DIRS["llama"])
    res_kimi = load_results(RESULT_DIRS["kimi"])
    res_llama = load_results(RESULT_DIRS["llama"])
    qtype_map = load_question_types_by_id(GRAPH_RAG_BENCH_JSONL)

    md = render_md(eval_kimi, eval_llama, res_kimi, res_llama, qtype_map)
    OUT_MD.write_text(md, encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())