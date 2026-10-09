"""Lexical test doubles for the three model protocols.

They are deterministic, need no downloads and behave sensibly on the synthetic demo
corpus, which is all the tests, the CI image smoke run and ``--backend demo`` ask of them.
They are not meant to verify real claims.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence

import numpy as np

from crag_verifier.text import NEGATIONS, content_tokens, raw_tokens

# Direction word stems: a premise using the opposite of the claim's direction flips it.
_OPPOSITE_STEMS = (
    ("increas", "decreas"),
    ("rais", "lower"),
    ("higher", "lower"),
    ("improv", "worsen"),
    ("promot", "inhibit"),
    ("shorten", "lengthen"),
    ("suppress", "enhanc"),
)


def _hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")


class HashingEncoder:
    """Signed feature hashing of unigrams and bigrams, L2-normalised."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"fake-hashing-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        toks = content_tokens(text)
        grams = toks + [a + "_" + b for a, b in itertools.pairwise(toks)]
        vec = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            h = _hash(g)
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_queries(texts)


def _coverage(hypothesis: str, premise: str) -> float:
    h = set(content_tokens(hypothesis))
    if not h:
        return 0.0
    return len(h & set(content_tokens(premise))) / len(h)


class OverlapCrossEncoder:
    """Logit from the share of query terms that occur in the passage."""

    name = "fake-overlap"

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        return np.array([10.0 * _coverage(q, p) - 5.0 for q, p in pairs], dtype=np.float32)


def _polarity(text: str) -> tuple[bool, list[str]]:
    tokens = raw_tokens(text)
    return any(t in NEGATIONS for t in tokens), tokens


def _flipped(hyp_words: list[str], prem_words: list[str]) -> bool:
    def has(words: list[str], prefix: str) -> bool:
        return any(w.startswith(prefix) for w in words)

    for a, b in _OPPOSITE_STEMS:
        h_a, h_b = has(hyp_words, a), has(hyp_words, b)
        p_a, p_b = has(prem_words, a), has(prem_words, b)
        if (h_a and p_b and not p_a) or (h_b and p_a and not p_b):
            return True
    return False


class LexicalNLI:
    """Coverage decides relatedness; negation or an opposite direction word flips the label."""

    name = "fake-lexical-nli"

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        for i, (premise, hypothesis) in enumerate(pairs):
            strength = _coverage(hypothesis, premise) ** 2
            neg_p, words_p = _polarity(premise)
            neg_h, words_h = _polarity(hypothesis)
            mismatch = (neg_p != neg_h) != _flipped(words_h, words_p)
            agree, disagree = (0.05, 0.95) if mismatch else (0.95, 0.05)
            out[i] = (agree * strength, 1.0 - strength, disagree * strength)
        return out
