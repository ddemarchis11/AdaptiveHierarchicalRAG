import sys
import json
import tiktoken
from pathlib import Path
from statistics import mean

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from cot_rag import CoTRAGConfig
    from config import PipelineConfig, DEFAULT_CONFIG
except ImportError as e:
    print(f"\nERRORE DI IMPORTAZIONE: {e}")
    sys.exit(1)

COSTS_MD = Path("token_costs.md")

STEP_SYSTEM_PROMPT = """You are a strategic retrieval controller for multi-hop reasoning.
Output ONLY valid JSON. Do not use tools.

You must update the scratchpad as SHORT BULLET FACTS grounded in the evidence.
Scratchpad format rules (strict):
- scratchpad_update MUST be a single STRING.
- Use 1-6 lines, each starting with '- '.
- Each line must end with a source id in parentheses, e.g. (Novel-xxxx_c_yy).
- Do NOT output JSON, dicts, YAML, code blocks, or nested structures inside scratchpad_update.
- Do NOT ask for information not present in the corpus unless evidence explicitly indicates such a value exists.

Entity bridging method:
1) Decompose the question into atomic sub-questions.
2) Extract salient entities from evidence.
3) If question links A->C and evidence supports A->B, treat B as a Bridge Entity and search B->C next.

Coverage / stop rules:
- enough=true ONLY if the current evidence supports all sub-questions at the level required by the question.
- If the evidence provides a qualitative relation that answers the question (e.g., 'near', 'in', 'by', 'toward'), that is sufficient unless the question explicitly requests a numeric/exact measure.
- If not enough, set enough=false and list missing as short phrases.
- When enough=false you MUST provide next_query (<= 8 words) that targets ONLY the missing part.
- next_query must include at least one specific entity from the evidence.
- next_query must NOT be a trivial reordering of a previous query in History.

Return ONLY a valid JSON object:
scratchpad_update (string), enough (boolean), missing (string or list), next_query (string)."""

STEP_USER_TEMPLATE = """Question:
{question}

Scratchpad:
{scratchpad}

History:
{history}

Current Evidence:
{evidence}
"""

FINAL_SYSTEM_PROMPT = "Answer using ONLY the provided evidence.\nIf evidence is insufficient for any part, say so."
FINAL_USER_TEMPLATE = "Question: {question}\n\nEvidence:\n{context}\n\n"

enc = tiktoken.get_encoding("cl100k_base")

def get_token_len(text: str) -> int:
    return len(enc.encode(text or ""))

COST_STEP_FIXED = get_token_len(STEP_SYSTEM_PROMPT) + get_token_len(STEP_USER_TEMPLATE)
COST_FINAL_FIXED = get_token_len(FINAL_SYSTEM_PROMPT) + get_token_len(FINAL_USER_TEMPLATE)
JSON_OVERHEAD = 40
HISTORY_ENTRY_AVG = 45

def analyze_directory(dir_path: Path, p_cfg: PipelineConfig, c_cfg: CoTRAGConfig):
    if not dir_path.exists():
        return None

    files = list(dir_path.glob("*.json"))
    if not files:
        return None

    stats = {
        "cot_input": [], "cot_output": [],
        "classic_input": [], "classic_output": [],
        "steps": [], "scratchpad": []
    }

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        q_len = get_token_len(data.get("question", ""))
        ans_len = get_token_len(data.get("llm_answer", ""))
        
        history = data.get("cot_history", [])
        scratchpad = data.get("cot_scratchpad", "")
        sp_len = get_token_len(scratchpad)
        n_steps = len(history)

        stats["steps"].append(n_steps)
        stats["scratchpad"].append(sp_len)
        stats["classic_output"].append(ans_len)

        avg_sp_step = sp_len / max(n_steps, 1)
        curr_sp_len = 0
        total_in = 0
        total_out = 0

        for i in range(n_steps):
            step_in = COST_STEP_FIXED + q_len + curr_sp_len
            step_in += min(i, c_cfg.history_window) * HISTORY_ENTRY_AVG
            step_in += c_cfg.top_k_step_context * p_cfg.chunking.chunk_size
            
            total_in += step_in
            total_out += avg_sp_step + JSON_OVERHEAD
            curr_sp_len += avg_sp_step

        final_in = COST_FINAL_FIXED + q_len
        final_in += c_cfg.top_k_final * p_cfg.chunking.chunk_size
        
        total_in += final_in
        total_out += ans_len

        stats["cot_input"].append(total_in)
        stats["cot_output"].append(total_out)

        classic_in = COST_FINAL_FIXED + q_len
        classic_in += p_cfg.query_top_k_chunks * p_cfg.chunking.chunk_size
        stats["classic_input"].append(classic_in)

    return {k: mean(v) for k, v in stats.items() if v}

def render_md_table(name, d) -> str:
    if not d:
        return f"**{name}**: No data found.\n"
    
    cot_tot = d['cot_input'] + d['cot_output']
    classic_tot = d['classic_input'] + d['classic_output']
    factor_tot = cot_tot / classic_tot if classic_tot > 0 else 0

    lines = []
    lines.append(f"## {name} Token Analysis")
    lines.append("")
    lines.append("| Metric | CoT (Avg) | Classic (Avg) | Factor |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Input Tokens | {d['cot_input']:.0f} | {d['classic_input']:.0f} | {d['cot_input']/d['classic_input']:.1f}x |")
    lines.append(f"| Output Tokens | {d['cot_output']:.0f} | {d['classic_output']:.0f} | {d['cot_output']/d['classic_output']:.1f}x |")
    lines.append(f"| **Total Tokens** | **{cot_tot:.0f}** | **{classic_tot:.0f}** | **{factor_tot:.2f}x** |")
    lines.append("")
    lines.append(f"- **Avg Steps**: {d['steps']:.2f}")
    lines.append(f"- **Avg Scratchpad**: {d['scratchpad']:.0f} tokens")
    lines.append("")
    return "\n".join(lines)

if __name__ == "__main__":
    dirs = {
        "Kimi": Path("llm_results_kimi"),
        "Llama": Path("llm_results_llama")
    }
    
    pipeline_cfg = DEFAULT_CONFIG
    cot_cfg = CoTRAGConfig()

    output_lines = ["# Token Cost Analysis: CoT vs Classic RAG\n"]

    for name, path in dirs.items():
        abs_path = Path(__file__).parent / path
        res = analyze_directory(abs_path, pipeline_cfg, cot_cfg)
        
        print(f"Analizzato {name}...")
        
        chunk = render_md_table(name, res)
        output_lines.append(chunk)

    COSTS_MD.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nReport salvato in: {COSTS_MD.resolve()}")