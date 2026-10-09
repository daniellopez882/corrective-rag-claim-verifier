from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crag_verifier.data import Document
from crag_verifier.strips import decompose, rationale_ids, recompose, select

words = st.sampled_from(
    ["cells", "were", "treated", "with", "insulin", "METHODS", "results", "dose"]
)
sentence = st.lists(words, min_size=1, max_size=9).map(" ".join)
documents = st.lists(sentence, min_size=1, max_size=40).map(
    lambda sents: Document(7, "title", tuple(sents))
)


@settings(max_examples=150, deadline=None)
@given(doc=documents, min_words=st.integers(1, 6), cap=st.integers(1, 64))
def test_decomposition_preserves_sentence_ids(doc: Document, min_words: int, cap: int) -> None:
    strips = decompose(doc, min_words=min_words, max_sentences=cap)
    covered = [sid for s in strips for sid in s.sentence_ids]
    assert covered == list(range(min(len(doc.sentences), cap)))
    for s in strips:
        assert s.doc_id == 7
        assert s.text == " ".join(doc.sentences[i].strip() for i in s.sentence_ids)
        # only the closing sentence of a strip may be long; merged ones are short fragments
        assert all(len(doc.sentences[i].split()) < min_words for i in s.sentence_ids[:-1])
    anchors = [s.anchor for s in strips]
    assert anchors == sorted(set(anchors))


def test_headers_merge_into_the_next_sentence() -> None:
    doc = Document(1, "t", ("BACKGROUND", "Insulin lowered glucose in mice.", "Short end"))
    strips = decompose(doc, min_words=4)
    assert [s.sentence_ids for s in strips] == [(0, 1), (2,)]
    assert strips[0].anchor == 1


@settings(max_examples=100, deadline=None)
@given(
    doc=documents, seed=st.integers(0, 999), threshold=st.floats(-2, 2), budget=st.integers(1, 5)
)
def test_selection_respects_threshold_budget_and_order(
    doc: Document, seed: int, threshold: float, budget: int
) -> None:
    strips = decompose(doc)
    scores = np.random.default_rng(seed).normal(size=len(strips))
    kept = select(strips, scores, threshold, max_sentences=budget)
    assert len(kept) <= budget
    assert [s.anchor for s in kept] == sorted(s.anchor for s in kept)
    kept_scores = [scores[strips.index(s)] for s in kept]
    assert all(v >= threshold for v in kept_scores)
    dropped = [scores[i] for i, s in enumerate(strips) if s not in kept and scores[i] >= threshold]
    if dropped and kept_scores:
        assert len(kept) == budget and max(dropped) <= min(kept_scores)
    assert rationale_ids(kept) == sorted(s.anchor for s in kept)


def test_recompose_restores_document_order() -> None:
    doc = Document(
        3, "t", ("First claim sentence here.", "Second one is here.", "Third one closes it.")
    )
    strips = decompose(doc)
    assert recompose([strips[2], strips[0]]) == "First claim sentence here. Third one closes it."


def test_select_requires_one_score_per_strip() -> None:
    strips = decompose(Document(1, "t", ("One full sentence here.",)))
    with pytest.raises(ValueError):
        select(strips, np.array([0.1, 0.2]), 0.0)
