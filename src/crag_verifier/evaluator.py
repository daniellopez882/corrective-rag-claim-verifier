"""The retrieval evaluator: features, a logistic scorer and a calibrator.

It answers one question per claim: how likely is it that a gold abstract is among the
top-3 retrieved? Its calibrated answer drives the three-way action triage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crag_verifier.calibration import (
    Calibrator,
    LogisticModel,
    calibrator_from_dict,
    fit_calibrator,
    log_loss,
)
from crag_verifier.models import CONTRADICT, ENTAIL
from crag_verifier.retrieval import RetrievalResult

FEATURES: tuple[str, ...] = (
    "ce_top1",
    "ce_gap12",
    "bm25_dense_agreement",
    "nli_max_entail",
    "nli_max_contradict",
)


def evaluator_features(
    result: RetrievalResult,
    sentence_probs: dict[int, np.ndarray],
    top_n: int = 3,
    agreement_k: int = 10,
) -> np.ndarray:
    """Feature vector in ``FEATURES`` order.

    ``sentence_probs`` maps each top-n doc id to its (n_sentences, 3) NLI matrix with the
    sentence as premise and the claim as hypothesis.
    """
    ce = result.ce_scores()
    top1 = float(ce[0]) if len(ce) else 0.0
    gap = float(ce[0] - ce[1]) if len(ce) > 1 else 0.0
    bm25 = set(result.bm25_top[:agreement_k])
    dense = set(result.dense_top[:agreement_k])
    agreement = len(bm25 & dense) / agreement_k
    entail = contradict = 0.0
    for doc_id in result.top(top_n):
        probs = sentence_probs.get(doc_id)
        if probs is not None and len(probs):
            entail = max(entail, float(probs[:, ENTAIL].max()))
            contradict = max(contradict, float(probs[:, CONTRADICT].max()))
    return np.array([top1, gap, agreement, entail, contradict], dtype=np.float64)


def gold_in_top(result: RetrievalResult, gold_doc_ids: set[int], top_n: int = 3) -> int:
    return int(bool(gold_doc_ids & set(result.top(top_n))))


def stratified_folds(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold id per sample, each class spread round-robin over folds after a shuffle."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=np.int64)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def out_of_fold_scores(
    x: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0, l2: float = 1.0
) -> np.ndarray:
    """Raw logistic scores for every sample from a model that never saw it."""
    y = np.asarray(y)
    k = max(2, min(k, int(y.sum()), int(len(y) - y.sum())))
    folds = stratified_folds(y, k, seed)
    scores = np.empty(len(y), dtype=np.float64)
    for f in range(k):
        test = folds == f
        model = LogisticModel.fit(x[~test], y[~test], l2=l2)
        scores[test] = model.decision_function(x[test])
    return scores


CALIBRATION_METHODS: tuple[str, ...] = ("platt", "isotonic")


def select_calibration(
    oof_scores: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 1
) -> tuple[str, dict[str, float]]:
    """Pick Platt or isotonic by held-out log-loss on the out-of-fold scores (train only)."""
    y = np.asarray(y)
    k = max(2, min(k, int(y.sum()), int(len(y) - y.sum())))
    folds = stratified_folds(y, k, seed)
    losses: dict[str, float] = {}
    for method in CALIBRATION_METHODS:
        held_out = np.empty(len(y), dtype=np.float64)
        for f in range(k):
            test = folds == f
            held_out[test] = fit_calibrator(method, oof_scores[~test], y[~test]).predict(
                oof_scores[test]
            )
        losses[method] = log_loss(held_out, y)
    return min(losses, key=lambda m: (losses[m], m)), losses


@dataclass
class RetrievalEvaluator:
    model: LogisticModel
    calibrator: Calibrator

    def raw_scores(self, x: np.ndarray) -> np.ndarray:
        return self.model.decision_function(x)

    def confidence(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(self.raw_scores(x))

    @classmethod
    def fit(
        cls, x: np.ndarray, y: np.ndarray, method: str, k: int = 5, seed: int = 0
    ) -> tuple[RetrievalEvaluator, np.ndarray]:
        """Cross-validated calibration: the calibrator sees only out-of-fold scores.

        Returns the evaluator (scorer refit on all rows) and the out-of-fold calibrated
        probabilities, which the threshold fit uses.
        """
        oof = out_of_fold_scores(x, y, k=k, seed=seed)
        calibrator = fit_calibrator(method, oof, y)
        return cls(LogisticModel.fit(x, y), calibrator), calibrator.predict(oof)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(FEATURES),
            "model": self.model.to_dict(),
            "calibration": self.calibrator.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetrievalEvaluator:
        if tuple(d["features"]) != FEATURES:
            raise ValueError(f"policy was fitted on features {d['features']}, expected {FEATURES}")
        return cls(LogisticModel.from_dict(d["model"]), calibrator_from_dict(d["calibration"]))
