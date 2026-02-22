import math, statistics, json
from dataclasses import dataclass
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from syntok import segmenter

MAX_TOKENS_EMBEDDER = 512

@dataclass
class Chunk:
    text: str
    length: int  
    rel_id: int


def split_into_sentences(
    text: str,
    tokenizer
) -> Tuple[List[str], List[int]]:
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

def mad(values: List[int]) -> float:
    if not values:
        return 0.0
    m = statistics.median(values)
    abs_dev = [abs(v - m) for v in values]
    return statistics.median(abs_dev)

def build_qa_text_list(filepath: str) -> Tuple[List, str]:
    with open(filepath, "r", encoding="utf-8") as f:
        doc = json.load(f)

    full_text = doc["context"]

    qa_list = []

    for qa in doc["qas"]:
        question = qa["question"]

        if qa["answers"]:
            answer_text = qa["answers"][0]["text"]
            answer_start = qa["answers"][0]["answer_start"]
        else:
            answer_text = None
            answer_start = None

        qa_list.append({
            "question": question,
            "answer_text": answer_text,
            "answer_start": answer_start,
        })
    return qa_list, full_text

class FixedSentenceChunker:
    def __init__(
        self,
        model: SentenceTransformer,
        overlap_pct: float = 0.1,
        min_sent_per_chunk: int = 2,
        max_sents_global: int = 50,
    ) -> None:
        self.model = model
        self.tokenizer = model.tokenizer
        self.overlap_pct = max(0.0, min(overlap_pct, 0.99))
        self.min_sent_per_chunk = max(1, min_sent_per_chunk)
        self.max_sents_global = max_sents_global

    def _compute_window_in_sentences(self, sent_lengths: List[int]) -> int:
        if not sent_lengths:
            return 0

        med = statistics.median(sent_lengths) or 1
        mean = statistics.mean(sent_lengths) or med

        if med <= 0:
            med = max(1, mean)

        window_sents_max = max(1, int(MAX_TOKENS_EMBEDDER / med))

        mad_val = mad(sent_lengths)
        if med < 1e-6:
            cv_robust = 0.0
        else:
            # 1.4826 ~ fattore di scala per rendere MAD approssimabile alla std
            approx_std = 1.4826 * mad_val
            cv_robust = approx_std / med

        cv_clamped = max(0.0, min(cv_robust, 1.0))

        alpha = math.exp(-cv_clamped)

        window_sents = int(window_sents_max * alpha)

        window_sents = max(self.min_sent_per_chunk, window_sents)
        window_sents = min(window_sents, self.max_sents_global)

        return window_sents


    def chunk(self, text: str) -> List[Chunk]:
        sentences, lengths = split_into_sentences(text, self.tokenizer)
        n = len(sentences)
        if n == 0:
            return []

        window_sents = self._compute_window_in_sentences(lengths)
        if window_sents <= 0:
            return []

        step_sents = int(window_sents * (1.0 - self.overlap_pct))
        if step_sents <= 0:
            step_sents = 1

        chunks: List[Chunk] = []
        start = 0
        rel_id = 1

        while start < n:
            logical_end = min(n, start + window_sents)

            total_tokens = 0
            final_end = start

            for i in range(start, logical_end):
                if total_tokens + lengths[i] > MAX_TOKENS_EMBEDDER:
                    break
                total_tokens += lengths[i]
                final_end = i + 1

            if final_end == start:
                sentence_text = sentences[start]
                tokenized = self.tokenizer(
                    sentence_text,
                    truncation=True,
                    max_length=MAX_TOKENS_EMBEDDER,
                    add_special_tokens=False,
                )
                token_ids = tokenized["input_ids"]
                chunk_text = self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True
                )
                token_len = len(token_ids)

                chunks.append(Chunk(text=chunk_text, length=token_len, rel_id=rel_id))
                rel_id += 1

                start += 1
                continue

            chunk_sents = sentences[start:final_end]
            chunk_text = " ".join(chunk_sents).strip()
            token_len = total_tokens 

            chunks.append(Chunk(text=chunk_text, length=token_len, rel_id=rel_id))
            rel_id += 1

            if final_end >= n:
                break

            start += step_sents

        return chunks


def fixed_token_chunking_smart(
    text: str,
    model: SentenceTransformer,
    overlap_pct: float = 0.2,
    min_sent_per_chunk: int = 2,
) -> List[Chunk]:
    chunker = FixedSentenceChunker(
        model=model,
        overlap_pct=overlap_pct,
        min_sent_per_chunk=min_sent_per_chunk,
    )
    return chunker.chunk(text)

def print_chunks(chunks: List[Chunk]):
    for ch in chunks:
        print(f"Chunk {ch.rel_id} – {ch.length} token")
        print(ch.text)
        print("-" * 80)