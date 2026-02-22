# Retrieval evaluation (Hit@k)

- Timestamp: `2026-02-19 17:18:23`
- QAs file: `subset_qas_only.jsonl` (filtered: `table_only == true`)
- ES URL: `http://localhost:9200`
- TOPK_SPARSE: `100`, TOPK_DENSE: `100`, RRF_K: `60`
- Dense model: `BAAI/bge-large-en-v1.5` (dims=1024)

## Summary

| Index | Mode | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | Avg Lat (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybridqa_tables_llm | sparse | 1463 | 0.312 | 0.458 | 0.528 | 0.616 | 0.690 | 0.414 | 10.4 |
| hybridqa_tables_llm | dense | 1463 | 0.493 | 0.683 | 0.751 | 0.840 | 0.912 | 0.615 | 9.2 |
| hybridqa_tables_llm | hybrid | 1463 | 0.422 | 0.597 | 0.670 | 0.757 | 0.840 | 0.534 | 0.1 |
| hybridqa_rows | sparse | 1463 | 0.402 | 0.455 | 0.471 | 0.493 | 0.553 | 0.442 | 10.4 |
| hybridqa_rows | dense | 1463 | 0.486 | 0.543 | 0.571 | 0.614 | 0.671 | 0.533 | 9.9 |
| hybridqa_rows | hybrid | 1463 | 0.501 | 0.560 | 0.593 | 0.645 | 0.709 | 0.546 | 0.2 |
| hybridqa_intros | sparse | 1463 | 0.185 | 0.292 | 0.340 | 0.411 | 0.494 | 0.261 | 9.8 |
| hybridqa_intros | dense | 1463 | 0.353 | 0.528 | 0.599 | 0.687 | 0.784 | 0.470 | 9.0 |
| hybridqa_intros | hybrid | 1463 | 0.265 | 0.396 | 0.465 | 0.561 | 0.637 | 0.357 | 0.2 |

## Notes

- Hybrid uses Reciprocal Rank Fusion (RRF) of sparse and dense results.
- Latency for hybrid is the RRF fusion overhead only (sparse + dense already measured separately).
