# LLM Evaluation Comparison (Kimi vs Llama)

- Paired (by ID) evaluation items: **323**
- GraphRAG-Bench question type mapping coverage: **323/323** mapped

## Overall judging score statistics

| metric | kimi mean | kimi stdev | llama mean | llama stdev | Δ (llama - kimi) mean | Δ stdev |
|---|---:|---:|---:|---:|---:|---:|
| `answer_correctness` | 0.785 | 0.389 | 0.624 | 0.426 | -0.161 | 0.430 |
| `faithfulness_to_context` | 0.878 | 0.314 | 0.661 | 0.419 | -0.217 | 0.423 |
| `evidence_coverage` | 0.776 | 0.395 | 0.604 | 0.421 | -0.172 | 0.437 |

## Judging scores by question type

| question_type | n | metric | kimi mean | llama mean | Δ (llama - kimi) mean |
|---|---:|---|---:|---:|---:|
| `Complex Reasoning` | 99 | `answer_correctness` | 0.808 | 0.677 | -0.131 |
| `Complex Reasoning` | 99 | `faithfulness_to_context` | 0.864 | 0.692 | -0.172 |
| `Complex Reasoning` | 99 | `evidence_coverage` | 0.798 | 0.667 | -0.131 |
| `Contextual Summarize` | 58 | `answer_correctness` | 0.888 | 0.776 | -0.112 |
| `Contextual Summarize` | 58 | `faithfulness_to_context` | 0.966 | 0.724 | -0.241 |
| `Contextual Summarize` | 58 | `evidence_coverage` | 0.879 | 0.707 | -0.172 |
| `Creative Generation` | 10 | `answer_correctness` | 1.000 | 0.500 | -0.500 |
| `Creative Generation` | 10 | `faithfulness_to_context` | 0.950 | 0.400 | -0.550 |
| `Creative Generation` | 10 | `evidence_coverage` | 0.950 | 0.500 | -0.450 |
| `Fact Retrieval` | 156 | `answer_correctness` | 0.718 | 0.542 | -0.176 |
| `Fact Retrieval` | 156 | `faithfulness_to_context` | 0.849 | 0.635 | -0.215 |
| `Fact Retrieval` | 156 | `evidence_coverage` | 0.712 | 0.532 | -0.179 |

## CoT step statistics

- Paired (by ID) results items: **323**

| scope | kimi mean steps | llama mean steps | Δ (llama - kimi) mean |
|---|---:|---:|---:|
| overall | 1.632 | 1.808 | 0.176 |

### Steps by question type

| question_type | n | kimi mean steps | llama mean steps | Δ (llama - kimi) mean |
|---|---:|---:|---:|---:|
| `Complex Reasoning` | 99 | 1.667 | 1.677 | 0.010 |
| `Contextual Summarize` | 58 | 1.741 | 1.672 | -0.069 |
| `Creative Generation` | 10 | 1.800 | 1.700 | -0.100 |
| `Fact Retrieval` | 156 | 1.558 | 1.949 | 0.391 |
