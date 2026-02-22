import numpy as np
import statistics
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from syntok import segmenter
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    length: int
    rel_id: int

MAX_TOKENS_EMBEDDER = 512

def split_into_sentences(text: str, tokenizer) -> Tuple[List[str], List[int]]:
    sentences: List[str] = []
    lens: List[int] = []

    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            sentence_text = "".join(str(token) for token in sentence).strip()
            if sentence_text:
                tokenized = tokenizer(
                    sentence_text,
                    truncation=False,
                    add_special_tokens=False,
                )
                tokens = tokenized["input_ids"]
                sentences.append(sentence_text)
                lens.append(len(tokens))

    return sentences, lens

def mad(values):
    if not values:
        return 0
    m = statistics.median(values)
    abs_dev = [abs(v - m) for v in values]
    return statistics.median(abs_dev)

def merge_short_sentences(sentences, lengths):
    if not sentences:
        return []
    
    med = statistics.median(lengths)
    mad_val = mad(lengths) or 1

    merged = []
    i = 0
    n = len(sentences)

    while i < n:
        curr_sent = sentences[i]
        curr_len = lengths[i]

        if curr_len >= med + mad_val:
            merged.append(curr_sent)
            i += 1
            continue

        prev_len = lengths[i - 1] if i - 1 >= 0 else None
        next_len = lengths[i + 1] if i + 1 < n else None

        next_under = next_len is not None and next_len < med

        if next_under:
            buffer_sent = curr_sent
            buffer_len = curr_len
            j = i + 1
            while j < n and buffer_len < med + mad_val:
                buffer_sent = buffer_sent + " " + sentences[j]
                buffer_len += lengths[j]
                j += 1
            merged.append(buffer_sent)
            i = j
        else:
            if prev_len is None and next_len is None:
                merged.append(curr_sent)
            elif prev_len is None:
                new_sent = curr_sent + " " + sentences[i + 1] if i + 1 < n else curr_sent
                merged.append(new_sent)
                i += 2
                continue
            elif next_len is None:
                if merged:
                    merged[-1] = merged[-1] + " " + curr_sent
                else:
                    merged.append(curr_sent)
            else:
                if prev_len <= next_len:
                    merged[-1] = merged[-1] + " " + curr_sent
                else:
                    new_sent = curr_sent + " " + sentences[i + 1]
                    merged.append(new_sent)
                    i += 2
                    continue
            i += 1
    return merged    

def three_window_combination(sentences: List[str]) -> List[str]:
    combined = []
    for i in range(1, len(sentences) - 1):
        combined.append(sentences[i-1] + " " + sentences[i] + " " + sentences[i+1])
    return combined

def get_distances(combined_sentences: List[str], model: SentenceTransformer):
    embeddings = model.encode(combined_sentences, normalize_embeddings=True)
    distances = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i+1]))
        distances.append(1 - sim)
    return distances

def choose_percentile_smooth(distances: np.ndarray,
                             cv_min=0.0, cv_max=0.5,
                             p_high=90.0, p_low=75.0) -> float:
    if len(distances) == 0:
        return 80.0

    mean_val = float(np.mean(distances))
    std_val = float(np.std(distances))

    if mean_val < 1e-6:
        return p_high

    cv = std_val / mean_val
    cv = max(cv_min, min(cv, cv_max))

    t = (cv - cv_min) / (cv_max - cv_min)
    percentile = p_high + t * (p_low - p_high)
    return percentile

def build_chunk(chunk_sents: List[str], model: SentenceTransformer, i: int) -> Tuple[Chunk, int]:
    chunk_raw = " ".join(chunk_sents)
    tokens = model.tokenizer(chunk_raw, truncation=False)["input_ids"]
    token_len = len(tokens)
    return Chunk(text=chunk_raw, length=token_len, rel_id=i), token_len

def semantic_chunking(
    text: str,
    model: SentenceTransformer,
    min_sent_per_chunk: int = 2
) -> Tuple[List[Chunk], List[int]]:
    tokenizer = model.tokenizer
    sentences_raw, lens = split_into_sentences(text, tokenizer)
    sentences_processed = merge_short_sentences(sentences_raw, lens)
    
    if len(sentences_processed) <= 1:
        chunk, length = build_chunk(sentences_processed if sentences_processed else [text], model, 1)
        return [chunk], [length]

    combined_sentences = three_window_combination(sentences_processed)

    if len(combined_sentences) < 2:
        chunk, length = build_chunk(sentences_processed, model, 1)
        return [chunk], [length]

    distances = get_distances(combined_sentences, model)
    percentile = choose_percentile_smooth(np.array(distances))
    thresh = np.percentile(distances, percentile)

    candidate_bps = [i + 1 for i, d in enumerate(distances) if d > thresh]
    breakpoints = sorted(set(candidate_bps + [len(sentences_processed) - 1]))

    chunks: List[Chunk] = []
    lengths: List[int] = []
    start = 0
    rel_id = 1
    
    for bp in breakpoints:
        if (bp - start + 1) < min_sent_per_chunk and bp != breakpoints[-1]:
            continue
        
        chunk_sents = sentences_processed[start:bp+1]
        chunk, length = build_chunk(chunk_sents, model, rel_id)
        chunks.append(chunk)
        lengths.append(length)

        rel_id += 1
        start = bp + 1

    return chunks, lengths

def print_chunks(chunks: List[Chunk]):
    for chunk in chunks:
        print(f"Chunk {chunk.rel_id} (tokens: {chunk.length}):")
        print(chunk.text)
        print("-" * 50)