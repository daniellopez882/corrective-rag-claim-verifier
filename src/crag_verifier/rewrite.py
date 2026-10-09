"""Query rewriting for the AMBIGUOUS action."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from crag_verifier.providers import LLMProvider, ProviderError
from crag_verifier.text import STOPWORDS, content_tokens

# Generic verbs of scientific claims. They carry the direction of a claim, which matters
# for NLI, but not its topic, which is what re-retrieval is looking for.
CLAIM_VERBS: frozenset[str] = frozenset(
    """
    show shows showed shown increase increases increased decrease decreases decreased
    reduce reduces reduced cause causes caused lead leads led result results resulted
    associated correlated correlates linked related affect affects affected promote
    promotes promoted inhibit inhibits inhibited improve improves improved induce induces
    induced enhance enhances enhanced prevent prevents prevented raise raises raised lower
    lowers lowered higher greater less
    """.split()
)

_SPLIT = re.compile(r"[^A-Za-z0-9\-]+")


class QueryRewriter(Protocol):
    name: str

    def rewrite(self, claim: str) -> str: ...


def candidate_phrases(claim: str) -> list[list[str]]:
    """Maximal runs of words that are neither stopwords nor generic claim verbs."""
    phrases: list[list[str]] = []
    current: list[str] = []
    for chunk in re.split(r"[,;:()\[\]]", claim):
        for word in _SPLIT.split(chunk):
            if not word:
                continue
            low = word.lower()
            if low in STOPWORDS or low in CLAIM_VERBS:
                if current:
                    phrases.append(current)
                current = []
            else:
                current.append(word)
        if current:
            phrases.append(current)
        current = []
    return phrases


class KeyphraseRewriter:
    """RAKE-style keyphrases scored by the summed BM25 IDF of their words.

    The rewritten query is the top phrases in their original order, which strips the
    assertion ("X reduces the risk of Y") down to its topic ("X risk Y").
    """

    name = "keyphrase"

    def __init__(self, idf: Callable[[str], float], max_phrases: int = 3):
        self.idf = idf
        self.max_phrases = max_phrases

    def score(self, phrase: list[str]) -> float:
        return sum(self.idf(t) for word in phrase for t in content_tokens(word))

    def rewrite(self, claim: str) -> str:
        phrases = candidate_phrases(claim)
        if not phrases:
            return claim
        ranked = sorted(range(len(phrases)), key=lambda i: (-self.score(phrases[i]), i))
        keep = sorted(ranked[: self.max_phrases])
        return " ".join(" ".join(phrases[i]) for i in keep)


REWRITE_SYSTEM = (
    "You turn a scientific claim into a short search query for a corpus of biomedical "
    "abstracts. Reply with the query only: key entities and concepts, no verbs of "
    "direction, at most twelve words."
)


class LLMRewriter:
    """Provider-backed rewrite; falls back to keyphrases if the provider fails."""

    def __init__(self, provider: LLMProvider, fallback: KeyphraseRewriter):
        self.provider = provider
        self.fallback = fallback
        self.name = f"llm:{provider.name}"

    def rewrite(self, claim: str) -> str:
        try:
            query = self.provider.complete(REWRITE_SYSTEM, f"Claim: {claim}", max_tokens=48)
        except ProviderError:
            return self.fallback.rewrite(claim)
        return query.splitlines()[0].strip() or self.fallback.rewrite(claim)
