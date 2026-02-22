# Benchmark Report: Semantic vs Fixed

| Parametro | Valore |
| :--- | :--- |
| **Modello Embedding** | `thenlper/gte-small` |
| **Fixed Window Size** | 256 |
| **Documenti Testati** | 1000 |
| **Query Totali** | 5000 |

## Analisi Comparativa

| Metrica | Semantic Chunking | Fixed Token Chunking |
| :--- | :--- | :--- |
| **Hit@1** | 0.8768 | 0.8706 |
| **Hit@2** | 0.9728 | 0.9680 |
| **Hit@3** | 0.9860 | 0.9800 |
| **Hit@4** | 0.9904 | 0.9860 |
| **Hit@5** | 0.9924 | 0.9902 |
