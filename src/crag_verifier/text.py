"""Tokenisation shared by BM25, keyphrase rewriting and the lexical test doubles."""

from __future__ import annotations

import re

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not of off on once
    only or other our ours ourselves out over own same she should so some such than that
    the their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you
    your yours yourself yourselves may might must shall via among within without upon per
    """.split()
)

# Negation is meaning-bearing for claims ("does not increase"), so it is kept as a token
# for the lexical NLI double even though BM25 drops it with the other stopwords.
NEGATIONS: frozenset[str] = frozenset({"no", "not", "never", "without", "nor", "cannot"})

_TOKEN = re.compile(r"[a-z0-9]+")


def raw_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def stem(token: str) -> str:
    """Strip common English plural endings; deliberately light so gene names survive."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stemmed tokens with stopwords removed (BM25 analyser)."""
    return [stem(t) for t in raw_tokens(text) if t not in STOPWORDS]
