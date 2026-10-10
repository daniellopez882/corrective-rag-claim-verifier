"""SciFact abstract- and sentence-level metrics, re-implemented per claim.

Semantics follow the official leaderboard evaluator (allenai/scifact-evaluator,
``evaluator/eval.py``): parity is tested on a fixture and re-checked against the real
script on every benchmark run. Counting per claim is what makes paired bootstrap
confidence intervals cheap: resample claims, sum their count vectors, recompute F1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from crag_verifier.data import Claim

MAX_ABSTRACT_SENTS = 3

COUNT_FIELDS: tuple[str, ...] = (
    "abstract_relevant",
    "abstract_retrieved",
    "abstract_correct_label_only",
    "abstract_correct_rationalized",
    "sentence_relevant",
    "sentence_retrieved",
    "sentence_correct_selection",
    "sentence_correct_label",
)

METRICS: dict[str, tuple[str, str, str]] = {
    # name: (relevant, retrieved, correct)
    "abstract_label_only": (
        "abstract_relevant",
        "abstract_retrieved",
        "abstract_correct_label_only",
    ),
    "abstract_rationalized": (
        "abstract_relevant",
        "abstract_retrieved",
        "abstract_correct_rationalized",
    ),
    "sentence_selection": ("sentence_relevant", "sentence_retrieved", "sentence_correct_selection"),
    "sentence_label": ("sentence_relevant", "sentence_retrieved", "sentence_correct_label"),
}

_IDX = {name: i for i, name in enumerate(COUNT_FIELDS)}


def claim_counts(claim: Claim, prediction: dict[str, Any]) -> np.ndarray:
    """Count vector (``COUNT_FIELDS`` order) for one claim's prediction."""
    c = np.zeros(len(COUNT_FIELDS), dtype=np.int64)
    c[_IDX["abstract_relevant"]] = len(claim.evidence)
    c[_IDX["sentence_relevant"]] = sum(
        len(r) for ev in claim.evidence.values() for r in ev.rationales
    )
    for doc_key, doc_pred in prediction.get("evidence", {}).items():
        label = doc_pred["label"]
        if label not in ("SUPPORT", "CONTRADICT"):
            raise ValueError(f"illegal predicted label {label!r}")
        sentences = list(doc_pred["sentences"])
        c[_IDX["abstract_retrieved"]] += 1
        c[_IDX["sentence_retrieved"]] += len(sentences)
        gold = claim.evidence.get(int(doc_key))
        if gold is None:
            continue
        gold_sets = [set(r) for r in gold.rationales]
        label_ok = label == gold.label
        if label_ok:
            c[_IDX["abstract_correct_label_only"]] += 1
            budget = max(MAX_ABSTRACT_SENTS, min(len(s) for s in gold_sets))
            truncated = set(sentences[:budget])
            if any(g.issubset(truncated) for g in gold_sets):
                c[_IDX["abstract_correct_rationalized"]] += 1
        predicted = set(sentences)
        n_correct = 0
        for sid in predicted:
            containing = [g for g in gold_sets if sid in g]
            if containing and containing[0].issubset(predicted):
                n_correct += 1
        c[_IDX["sentence_correct_selection"]] += n_correct
        if label_ok:
            c[_IDX["sentence_correct_label"]] += n_correct
    return c


def f1_scores(counts: np.ndarray) -> dict[str, float]:
    """Precision, recall and F1 for every metric from summed counts."""
    out: dict[str, float] = {}
    for name, (rel, ret, cor) in METRICS.items():
        relevant, retrieved, correct = (counts[_IDX[rel]], counts[_IDX[ret]], counts[_IDX[cor]])
        precision = correct / retrieved if retrieved else 0.0
        recall = correct / relevant if relevant else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"{name}_precision"] = float(precision)
        out[f"{name}_recall"] = float(recall)
        out[f"{name}_f1"] = float(f1)
    return out


def count_matrix(claims: Sequence[Claim], predictions: Sequence[dict[str, Any]]) -> np.ndarray:
    if [c.id for c in claims] != [p["id"] for p in predictions]:
        raise ValueError("predictions must list the same claims in the same order")
    return np.stack([claim_counts(c, p) for c, p in zip(claims, predictions, strict=True)])


def label_counts(counts: np.ndarray, claims: Sequence[int] | None = None) -> dict[str, int]:
    """Correct and wrong abstract labels in a count matrix, over the given rows (all if None).

    A wrong label is any labelled abstract that is not a correct one: a gold abstract with
    the other label, or an abstract that is not gold evidence for the claim.
    """
    rows = counts if claims is None else counts[list(claims)]
    correct = int(rows[:, _IDX["abstract_correct_label_only"]].sum())
    retrieved = int(rows[:, _IDX["abstract_retrieved"]].sum())
    return {"correct_labels": correct, "wrong_labels": retrieved - correct}


def evaluate(claims: Sequence[Claim], predictions: Sequence[dict[str, Any]]) -> dict[str, float]:
    return f1_scores(count_matrix(claims, predictions).sum(axis=0))


BOOTSTRAP_METRICS: tuple[str, ...] = (
    "abstract_label_only_f1",
    "abstract_rationalized_f1",
    "sentence_label_f1",
)


def _f1_rows(summed: np.ndarray) -> dict[str, np.ndarray]:
    """F1 of every metric for each row of summed count vectors."""
    out: dict[str, np.ndarray] = {}
    for name, (rel, ret, cor) in METRICS.items():
        relevant, retrieved, correct = (
            summed[:, _IDX[field]].astype(np.float64) for field in (rel, ret, cor)
        )
        zeros = np.zeros(len(summed))
        precision = np.divide(correct, retrieved, out=zeros.copy(), where=retrieved > 0)
        recall = np.divide(correct, relevant, out=zeros.copy(), where=relevant > 0)
        both = precision + recall
        out[f"{name}_f1"] = np.divide(2 * precision * recall, both, out=zeros, where=both > 0)
    return out


def bootstrap(
    counts: dict[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
    n_resamples: int = 10000,
    seed: int = 13,
    metrics: Sequence[str] = BOOTSTRAP_METRICS,
) -> dict[str, Any]:
    """Percentile 95% CIs per system, and paired CIs for each (system, baseline) difference.

    Every system is scored on the same resampled claim indices, so differences are paired.
    """
    n = next(iter(counts.values())).shape[0]
    rng = np.random.default_rng(seed)
    weights = np.empty((n_resamples, n), dtype=np.int64)  # times each claim is drawn
    for r in range(n_resamples):
        weights[r] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    values = {system: _f1_rows(weights @ matrix) for system, matrix in counts.items()}
    point = {system: f1_scores(matrix.sum(axis=0)) for system, matrix in counts.items()}

    def interval(v: np.ndarray) -> dict[str, float]:
        return {
            "ci95_low": float(np.percentile(v, 2.5)),
            "ci95_high": float(np.percentile(v, 97.5)),
        }

    differences: dict[str, Any] = {}
    for system, baseline in comparisons:
        entry: dict[str, Any] = {"system": system, "baseline": baseline}
        for m in metrics:
            d = values[system][m] - values[baseline][m]
            entry[m] = {
                "estimate": point[system][m] - point[baseline][m],
                **interval(d),
                "share_of_resamples_at_or_below_zero": float(np.mean(d <= 0.0)),
            }
        differences[f"{system} - {baseline}"] = entry
    return {
        "resamples": n_resamples,
        "seed": seed,
        "intervals": {s: {m: interval(values[s][m]) for m in metrics} for s in counts},
        "differences": differences,
    }
