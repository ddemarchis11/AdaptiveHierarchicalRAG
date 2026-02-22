from typing import List, Dict, Any, Optional, Callable, Tuple
from rank_bm25 import BM25Okapi
from syntok import segmenter


def default_tokenize(text: str) -> List[str]:
    toks: List[str] = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            for token in sentence:
                t = str(token).lower().strip()
                if t:
                    toks.append(t)
    return toks


class HybridRetriever:
    def __init__(
        self,
        chroma_collection,
        model,
        tokenize_fn: Callable[[str], List[str]] = default_tokenize,
        rrf_k: int = 60,
        w_dense: float = 1.0,
        w_sparse: float = 1.0,
        bm25_corpus_ids: Optional[List[str]] = None,
        bm25_corpus_texts: Optional[List[str]] = None,
        bm25_corpus_metas: Optional[List[dict]] = None,
    ):
        self.col = chroma_collection
        self.model = model
        self.tokenize = tokenize_fn
        self.rrf_k = rrf_k
        self.w_dense = w_dense
        self.w_sparse = w_sparse

        if bm25_corpus_texts is None or bm25_corpus_ids is None:
            data = self.col.get(include=["documents", "metadatas"])
            self.ids = data["ids"]
            self.texts = data["documents"]
            self.metas = data["metadatas"]
        else:
            self.ids = bm25_corpus_ids
            self.texts = bm25_corpus_texts
            self.metas = bm25_corpus_metas or [{} for _ in bm25_corpus_texts]

        tokenized_corpus = [self.tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.id_to_idx: Dict[str, int] = {cid: i for i, cid in enumerate(self.ids)}

        self._lookup: Dict[str, Any] = {
            cid: (txt, meta)
            for cid, txt, meta in zip(self.ids, self.texts, self.metas)
        }

    def _dense_search(self, query: str, k: int) -> List[str]:
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
        ).tolist()

        res = self.col.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        if not res or not res.get("ids"):
            return []
        return res["ids"][0]

    def _sparse_search(self, query: str, k: int) -> List[str]:
        q_tok = self.tokenize(query)
        scores = self.bm25.get_scores(q_tok)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.ids[i] for i in top_idx]

    def _rrf_fuse(
        self,
        dense_ids: List[str],
        sparse_ids: List[str],
        k_final: int,
    ) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}

        for r, cid in enumerate(dense_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + self.w_dense / (self.rrf_k + r)

        for r, cid in enumerate(sparse_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + self.w_sparse / (self.rrf_k + r)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:k_final]

    def retrieve(
        self,
        query: str,
        k_dense: int = 8,
        k_sparse: int = 8,
        k_final: int = 6,
    ) -> List[Dict[str, Any]]:
        dense_ids = self._dense_search(query, k_dense)
        sparse_ids = self._sparse_search(query, k_sparse)

        fused = self._rrf_fuse(dense_ids, sparse_ids, k_final)

        out: List[Dict[str, Any]] = []
        for cid, score in fused:
            if cid not in self._lookup:
                continue
            txt, meta = self._lookup[cid]
            out.append({"id": cid, "text": txt, "metadata": meta, "score": float(score)})
        return out

    def score(self, query: str, docs: List[Dict[str, Any]]) -> List[float]:
        q_tok = self.tokenize(query)
        all_scores = self.bm25.get_scores(q_tok)

        out_scores: List[float] = []
        for d in docs:
            cid = d.get("id") if isinstance(d, dict) else None
            if cid is not None and cid in self.id_to_idx:
                out_scores.append(float(all_scores[self.id_to_idx[cid]]))
            else:
                out_scores.append(0.0)

        return out_scores