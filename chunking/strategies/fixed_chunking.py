from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    length: int
    rel_id: int
      
from typing import List

def fixed_token_chunking(
    text: str,
    window_size: int,
    step: int | None = None,
    overlap_pct: float = 0.1,
) -> List[str]:

    if window_size <= 0:
        raise ValueError("window_size deve essere > 0")

    tokens = text.split()
    n_tokens = len(tokens)

    if n_tokens == 0:
        return []

    overlap_pct = max(0.0, min(overlap_pct, 0.99))

    if step is None:
        step = int(window_size * (1.0 - overlap_pct))
        if step <= 0:
            step = 1 

    chunks: List[str] = []

    start = 0
    while start < n_tokens:
        end = start + window_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break

        chunk_text = " ".join(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= n_tokens:
            break

        start += step

    return chunks
