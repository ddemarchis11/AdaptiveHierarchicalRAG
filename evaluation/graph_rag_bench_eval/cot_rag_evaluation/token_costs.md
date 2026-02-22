# Token Cost Analysis: CoT vs Classic RAG

## Kimi Token Analysis

| Metric | CoT (Avg) | Classic (Avg) | Factor |
|---|---:|---:|---:|
| Input Tokens | 5259 | 3060 | 1.7x |
| Output Tokens | 267 | 62 | 4.3x |
| **Total Tokens** | **5526** | **3122** | **1.77x** |

- **Avg Steps**: 1.72
- **Avg Scratchpad**: 136 tokens

## Llama Token Analysis

| Metric | CoT (Avg) | Classic (Avg) | Factor |
|---|---:|---:|---:|
| Input Tokens | 5382 | 3060 | 1.8x |
| Output Tokens | 1261 | 173 | 7.3x |
| **Total Tokens** | **6643** | **3234** | **2.05x** |

- **Avg Steps**: 1.57
- **Avg Scratchpad**: 1025 tokens
