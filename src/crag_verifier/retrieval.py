"""Hybrid first stage (own BM25 + dense), reciprocal-rank fusion and cross-encoder rerank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crag_verifier.bm25 import BM25Index, top_k_indices
from crag_verifier.data import Corpus
from crag_verifier.models import DenseEncoder, PairScorer
from crag_verifier.text import content_tokens


@dataclass(frozen=True)
class Candidate:
    doc_id: int
    ce_score: float


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    candidates: tuple[Candidate, ...]  # sorted by cross-encoder score, best first
    bm25_top: tuple[int, ...]  # first-stage doc ids, kept for the agreement feature
    dense_top: tuple[int, ...]

    def top(self, n: int) -> list[int]:
        return [c.doc_id for c in self.candidates[:n]]

    def ce_scores(self) -> np.ndarray:
        return np.array([c.ce_score for c in self.candidates], dtype=np.float64)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))


class HybridRetriever:
    def __init__(
        self,
        corpus: Corpus,
        encoder: DenseEncoder,
        cross_encoder: PairScorer,
        doc_vectors: np.ndarray,
        *,
        first_stage_k: int = 100,
        rerank_depth: int = 20,
        expand_depth: int = 10,
        rrf_k: int = 60,
    ):
        if doc_vectors.shape[0] != len(corpus):
            raise ValueError("one dense vector per corpus document is required")
        self.corpus = corpus
        self.encoder = encoder
        self.cross_encoder = cross_encoder
        self.doc_vectors = doc_vectors.astype(np.float32, copy=False)
        self.first_stage_k = first_stage_k
        self.rerank_depth = rerank_depth
        self.expand_depth = expand_depth
        self.rrf_k = rrf_k
        self.bm25 = BM25Index([content_tokens(d.text) for d in corpus])
        self._ids = [d.doc_id for d in corpus]

    def bm25_search(self, query: str, k: int) -> list[int]:
        return [self._ids[i] for i, _ in self.bm25.search(content_tokens(query), k)]

    def dense_search(self, query: str, k: int) -> list[int]:
        q = self.encoder.encode_queries([query])[0]
        sims = self.doc_vectors @ q.astype(np.float32)
        return [self._ids[i] for i in top_k_indices(sims, k)]

    def _rerank(self, claim: str, doc_ids: list[int], known: dict[int, float]) -> list[Candidate]:
        missing = [d for d in doc_ids if d not in known]
        scores = self.cross_encoder.score([(claim, self.corpus[d].text) for d in missing])
        merged = {**known, **{d: float(s) for d, s in zip(missing, scores, strict=True)}}
        position = {d: i for i, d in enumerate(doc_ids)}
        order = sorted(doc_ids, key=lambda d: (-merged[d], position[d]))
        return [Candidate(d, merged[d]) for d in order]

    def retrieve(self, claim: str) -> RetrievalResult:
        bm25 = self.bm25_search(claim, self.first_stage_k)
        dense = self.dense_search(claim, self.first_stage_k)
        pool = reciprocal_rank_fusion([bm25, dense], self.rrf_k)[: self.rerank_depth]
        return RetrievalResult(
            claim, tuple(self._rerank(claim, pool, {})), tuple(bm25), tuple(dense)
        )

    def expand(self, claim: str, rewritten: str, base: RetrievalResult) -> RetrievalResult:
        """Dense re-retrieval with the rewritten query, merged into the original pool.

        The merged pool is re-ranked against the original claim: the rewrite only widens
        the net, relevance is still judged on what the claim actually says.
        """
        fresh = self.dense_search(rewritten, self.expand_depth)
        known = {c.doc_id: c.ce_score for c in base.candidates}
        pool = list(known) + [d for d in fresh if d not in known]
        return RetrievalResult(
            rewritten, tuple(self._rerank(claim, pool, known)), base.bm25_top, base.dense_top
        )
