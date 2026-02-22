# LLM Evaluation Comparison (Kimi vs Llama)

- Paired (by ID) evaluation items: **350**
- GraphRAG-Bench question type mapping coverage: **350/350** mapped

## Overall judging score statistics

| metric | kimi mean | kimi stdev | llama mean | llama stdev | Δ (llama - kimi) mean | Δ stdev |
|---|---:|---:|---:|---:|---:|---:|
| `answer_correctness` | 0.731 | 0.427 | 0.556 | 0.473 | -0.176 | 0.455 |
| `faithfulness_to_context` | 0.903 | 0.284 | 0.667 | 0.463 | -0.236 | 0.497 |
| `evidence_coverage` | 0.720 | 0.428 | 0.541 | 0.471 | -0.179 | 0.451 |

## Judging scores by question type

| question_type | n | metric | kimi mean | llama mean | Δ (llama - kimi) mean |
|---|---:|---|---:|---:|---:|
| `bridge` | 280 | `answer_correctness` | 0.720 | 0.512 | -0.207 |
| `bridge` | 280 | `faithfulness_to_context` | 0.896 | 0.639 | -0.257 |
| `bridge` | 280 | `evidence_coverage` | 0.707 | 0.500 | -0.207 |
| `comparison` | 70 | `answer_correctness` | 0.779 | 0.729 | -0.050 |
| `comparison` | 70 | `faithfulness_to_context` | 0.929 | 0.779 | -0.150 |
| `comparison` | 70 | `evidence_coverage` | 0.771 | 0.707 | -0.064 |

## CoT step statistics

- Paired (by ID) results items: **350**

| scope | kimi mean steps | llama mean steps | Δ (llama - kimi) mean |
|---|---:|---:|---:|
| overall | 1.697 | 1.654 | -0.043 |

### Steps by question type

| question_type | n | kimi mean steps | llama mean steps | Δ (llama - kimi) mean |
|---|---:|---:|---:|---:|
| `bridge` | 280 | 1.704 | 1.639 | -0.064 |
| `comparison` | 70 | 1.671 | 1.714 | 0.043 |
