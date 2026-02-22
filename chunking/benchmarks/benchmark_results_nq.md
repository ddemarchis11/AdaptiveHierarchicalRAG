# Benchmark Report: Semantic vs Fixed Chunking

| Parametro | Valore |
| :--- | :--- |
| **Modello Embedding** | `thenlper/gte-small` |
| **Documenti Testati** | 1000 |
| **Query Totali** | 5000 |

## Accuratezza Retrieval (Hit@K)

| Metrica | Semantic Chunking | Fixed Token Chunking |
| :--- | :--- | :--- |
| **Avg Chunk Length (Top-1)** | 406.71 tokens | 478.22 tokens |
| **Hit@1** | 0.8768 | 0.8816 |
| **Hit@2** | 0.9728 | 0.9768 |
| **Hit@3** | 0.9860 | 0.9878 |
| **Hit@4** | 0.9904 | 0.9918 |
| **Hit@5** | 0.9924 | 0.9938 |

---
**Nota:** L'accuratezza Hit@K indica la percentuale di casi in cui la risposta corretta è contenuta in almeno uno dei primi K chunk recuperati.
