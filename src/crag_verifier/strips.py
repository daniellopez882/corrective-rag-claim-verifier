"""Decompose-then-recompose: split abstracts into knowledge strips, keep the relevant ones.

A strip is one sentence, except that fragments shorter than ``min_words`` (section
headers such as "METHODS", or "Results:") are merged into the sentence that follows, so
the relevance model never has to score a header on its own. Every strip records the
SciFact sentence ids it covers; its anchor is the full sentence that closed it, and the
anchors of the strips kept for a document are the rationale reported for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from crag_verifier.data import Document

MAX_RATIONALE_SENTENCES = 3  # the official metric truncates rationales to three sentences


@dataclass(frozen=True)
class Strip:
    doc_id: int
    sentence_ids: tuple[int, ...]
    text: str

    @property
    def anchor(self) -> int:
        """The sentence that closed the strip; merged headers before it are context only."""
        return self.sentence_ids[-1]


def decompose(doc: Document, min_words: int = 4, max_sentences: int = 64) -> list[Strip]:
    """Strips in document order covering sentences 0..min(n, max_sentences)-1 exactly once."""
    strips: list[Strip] = []
    pending_ids: list[int] = []
    pending_text: list[str] = []
    sentences = doc.sentences[:max_sentences]
    for sid, sentence in enumerate(sentences):
        pending_ids.append(sid)
        pending_text.append(sentence.strip())
        is_last = sid == len(sentences) - 1
        if len(sentence.split()) >= min_words or is_last:
            strips.append(Strip(doc.doc_id, tuple(pending_ids), " ".join(pending_text)))
            pending_ids, pending_text = [], []
    return strips


def select(
    strips: Sequence[Strip],
    relevance: np.ndarray,
    min_relevance: float,
    max_sentences: int = MAX_RATIONALE_SENTENCES,
) -> list[Strip]:
    """Most relevant strips at or above the threshold, at most ``max_sentences``, in doc order."""
    if len(strips) != len(relevance):
        raise ValueError("one relevance score per strip is required")
    chosen: list[int] = []
    for i in np.argsort(-np.asarray(relevance), kind="stable"):
        if relevance[i] < min_relevance or len(chosen) == max_sentences:
            break
        chosen.append(int(i))
    return [strips[i] for i in sorted(chosen)]


def recompose(selected: Sequence[Strip]) -> str:
    """Kept strips joined in their original order: the refined evidence passage."""
    ordered = sorted(selected, key=lambda s: s.sentence_ids[0])
    return " ".join(s.text for s in ordered)


def rationale_ids(selected: Sequence[Strip]) -> list[int]:
    return sorted(s.anchor for s in selected)
