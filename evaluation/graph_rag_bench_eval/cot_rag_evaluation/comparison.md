# LLM Evaluation Comparison (Kimi vs Llama)

- Paired (by ID) evaluation items: **323**
- GraphRAG-Bench question type mapping coverage: **323/323** mapped

## Overall judging score statistics

| metric | kimi mean | kimi stdev | llama mean | llama stdev | Δ (llama - kimi) mean | Δ stdev |
|---|---:|---:|---:|---:|---:|---:|
| `answer_correctness` | 0.771 | 0.388 | 0.633 | 0.428 | -0.138 | 0.402 |
| `faithfulness_to_context` | 0.881 | 0.298 | 0.675 | 0.414 | -0.206 | 0.411 |
| `evidence_coverage` | 0.762 | 0.394 | 0.611 | 0.424 | -0.150 | 0.409 |

## Judging scores by question type

| question_type | n | metric | kimi mean | llama mean | Δ (llama - kimi) mean |
|---|---:|---|---:|---:|---:|
| `Complex Reasoning` | 99 | `answer_correctness` | 0.823 | 0.712 | -0.111 |
| `Complex Reasoning` | 99 | `faithfulness_to_context` | 0.894 | 0.702 | -0.192 |
| `Complex Reasoning` | 99 | `evidence_coverage` | 0.803 | 0.687 | -0.116 |
| `Contextual Summarize` | 58 | `answer_correctness` | 0.888 | 0.828 | -0.060 |
| `Contextual Summarize` | 58 | `faithfulness_to_context` | 0.974 | 0.784 | -0.190 |
| `Contextual Summarize` | 58 | `evidence_coverage` | 0.888 | 0.767 | -0.121 |
| `Creative Generation` | 10 | `answer_correctness` | 0.750 | 0.600 | -0.150 |
| `Creative Generation` | 10 | `faithfulness_to_context` | 0.750 | 0.500 | -0.250 |
| `Creative Generation` | 10 | `evidence_coverage` | 0.750 | 0.550 | -0.200 |
| `Fact Retrieval` | 156 | `answer_correctness` | 0.696 | 0.513 | -0.183 |
| `Fact Retrieval` | 156 | `faithfulness_to_context` | 0.846 | 0.628 | -0.218 |
| `Fact Retrieval` | 156 | `evidence_coverage` | 0.689 | 0.510 | -0.179 |

## CoT step statistics

- Paired (by ID) results items: **323**

| scope | kimi mean steps | llama mean steps | Δ (llama - kimi) mean |
|---|---:|---:|---:|
| overall | 1.715 | 1.573 | -0.142 |

### Steps by question type

| question_type | n | kimi mean steps | llama mean steps | Δ (llama - kimi) mean |
|---|---:|---:|---:|---:|
| `Complex Reasoning` | 99 | 1.626 | 1.495 | -0.131 |
| `Contextual Summarize` | 58 | 1.897 | 1.534 | -0.362 |
| `Creative Generation` | 10 | 1.700 | 1.800 | 0.100 |
| `Fact Retrieval` | 156 | 1.705 | 1.622 | -0.083 |
