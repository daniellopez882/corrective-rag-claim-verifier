"""Okapi BM25 over an inverted index held in flat numpy arrays."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np


class BM25Index:
    """Lucene-style BM25: idf = ln(1 + (N - df + 0.5) / (df + 0.5)).

    Defaults k1=0.9, b=0.4 follow the Anserini settings commonly used on BEIR/SciFact.
    """

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 0.9, b: float = 0.4):
        if not documents:
            raise ValueError("BM25Index needs at least one document")
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avg_len = float(lengths.mean()) or 1.0
        self._length_norm = k1 * (1.0 - b + b * lengths / self.avg_len)

        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_idx, tokens in enumerate(documents):
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, []).append((doc_idx, tf))

        self.vocab: dict[str, int] = {}
        offsets = [0]
        doc_ids: list[int] = []
        tfs: list[int] = []
        for term_id, (term, plist) in enumerate(sorted(postings.items())):
            self.vocab[term] = term_id
            doc_ids.extend(p[0] for p in plist)
            tfs.extend(p[1] for p in plist)
            offsets.append(len(doc_ids))
        self._offsets = np.array(offsets, dtype=np.int64)
        self._doc_ids = np.array(doc_ids, dtype=np.int64)
        self._tfs = np.array(tfs, dtype=np.float64)
        df = np.diff(self._offsets).astype(np.float64)
        self._idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5))

    def idf(self, term: str) -> float:
        """IDF of a term; unseen terms get the maximum possible IDF (df = 0)."""
        term_id = self.vocab.get(term)
        if term_id is None:
            return math.log1p((self.n_docs + 0.5) / 0.5)
        return float(self._idf[term_id])

    def scores(self, query: Sequence[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        for term, qtf in Counter(query).items():
            term_id = self.vocab.get(term)
            if term_id is None:
                continue
            lo, hi = self._offsets[term_id], self._offsets[term_id + 1]
            docs = self._doc_ids[lo:hi]
            tf = self._tfs[lo:hi]
            gain = tf * (self.k1 + 1.0) / (tf + self._length_norm[docs])
            out[docs] += qtf * self._idf[term_id] * gain
        return out

    def search(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        """Top-k (doc index, score) with score > 0, ties broken by lower index."""
        scores = self.scores(query)
        return [(int(i), float(scores[i])) for i in top_k_indices(scores, k) if scores[i] > 0.0]


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first, ties broken by lower index.

    A plain argpartition picks an arbitrary subset of the documents tied at the k-th
    score, so every document tied with the k-th one is sorted before truncating.
    """
    n = len(scores)
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k < n:
        kth = np.partition(scores, n - k)[n - k]
        candidates = np.flatnonzero(scores >= kth)
    else:
        candidates = np.arange(n)
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:k]]
