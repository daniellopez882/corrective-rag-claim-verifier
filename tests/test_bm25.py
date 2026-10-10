from __future__ import annotations

import math
from collections import Counter

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from crag_verifier.bm25 import BM25Index, top_k_indices
from crag_verifier.text import content_tokens, stem

VOCAB = ["gene", "cell", "mouse", "tumor", "insulin", "protein", "risk", "dose"]
docs_strategy = st.lists(
    st.lists(st.sampled_from(VOCAB), min_size=0, max_size=12), min_size=1, max_size=15
)


def reference_scores(docs: list[list[str]], query: list[str], k1: float, b: float) -> np.ndarray:
    n = len(docs)
    avg = sum(len(d) for d in docs) / n or 1.0
    out = np.zeros(n)
    for term, qtf in Counter(query).items():
        df = sum(1 for d in docs if term in d)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(term)
            if tf:
                out[i] += qtf * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg))
    return out


@settings(max_examples=60, deadline=None)
@given(docs=docs_strategy, query=st.lists(st.sampled_from([*VOCAB, "absent"]), max_size=6))
def test_scores_match_the_textbook_formula(docs: list[list[str]], query: list[str]) -> None:
    index = BM25Index(docs, k1=0.9, b=0.4)
    np.testing.assert_allclose(
        index.scores(query), reference_scores(docs, query, 0.9, 0.4), rtol=1e-12
    )


@settings(max_examples=40, deadline=None)
@given(docs=docs_strategy, query=st.lists(st.sampled_from(VOCAB), min_size=1, max_size=4))
def test_search_is_sorted_and_positive(docs: list[list[str]], query: list[str]) -> None:
    index = BM25Index(docs)
    hits = index.search(query, k=5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)
    assert len({i for i, _ in hits}) == len(hits)


def test_ties_break_by_document_order() -> None:
    index = BM25Index([["gene"], ["gene"], ["cell"]])
    assert [i for i, _ in index.search(["gene"], k=3)] == [0, 1]


@settings(max_examples=200, deadline=None)
@given(
    scores=st.lists(st.integers(0, 3), min_size=1, max_size=60),
    k=st.integers(0, 70),
)
def test_top_k_keeps_the_lowest_indices_among_ties(scores: list[int], k: int) -> None:
    # Few distinct values, so most cut-offs fall inside a run of tied scores.
    expected = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    assert top_k_indices(np.array(scores, dtype=np.float64), k).tolist() == expected


def test_rarer_terms_get_higher_idf_and_unseen_terms_the_maximum() -> None:
    index = BM25Index([["gene", "cell"], ["gene"], ["gene", "tumor"]])
    assert index.idf("tumor") > index.idf("gene")
    assert index.idf("never-seen") >= index.idf("tumor")


def test_analyser_drops_stopwords_and_light_stems() -> None:
    assert content_tokens("The mice were given doses of IL-6") == [
        "mice",
        "given",
        "dose",
        "il",
        "6",
    ]
    assert stem("studies") == "study"
    assert stem("analysis") == "analysis"
