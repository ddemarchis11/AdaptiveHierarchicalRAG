# Benchmark Report: Semantic vs Fixed Chunking

| Parametro | Valore |
| :--- | :--- |
| **Modello Embedding** | `thenlper/gte-small` |
| **Documenti Testati** | 1000 |
| **Query Totali** | 5825 |

## Accuratezza Retrieval (Hit@K)

| Metrica | Semantic Chunking | Fixed Token Chunking |
| :--- | :--- | :--- |
| **Avg Chunk Length (Top-1)** | 148.13 tokens | 150.02 tokens |
| **Hit@1** | 0.9888 | 0.9906 |
| **Hit@2** | 1.0000 | 1.0000 |
| **Hit@3** | 1.0000 | 1.0000 |
| **Hit@4** | 1.0000 | 1.0000 |
| **Hit@5** | 1.0000 | 1.0000 |

---
**Nota:** L'accuratezza Hit@K indica la percentuale di casi in cui la risposta corretta è contenuta in almeno uno dei primi K chunk recuperati.
