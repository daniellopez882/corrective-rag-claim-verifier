from __future__ import annotations

import numpy as np
import pytest

from crag_verifier.evaluator import (
    FEATURES,
    RetrievalEvaluator,
    evaluator_features,
    gold_in_top,
    out_of_fold_scores,
    select_calibration,
    stratified_folds,
)
from crag_verifier.retrieval import Candidate, RetrievalResult, reciprocal_rank_fusion


def result() -> RetrievalResult:
    return RetrievalResult(
        "claim",
        (Candidate(5, 4.0), Candidate(2, 1.5), Candidate(9, -1.0), Candidate(7, -3.0)),
        bm25_top=(5, 2, 1, 3, 4, 6, 8, 10, 11, 12, 99),
        dense_top=(2, 5, 9, 13, 14, 15, 16, 17, 18, 1),
    )


def test_features_follow_their_definitions() -> None:
    probs = {
        5: np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]]),
        2: np.array([[0.1, 0.1, 0.8]]),
        9: np.zeros((0, 3)),
        7: np.array([[0.99, 0.0, 0.01]]),  # not in the top 3: must be ignored
    }
    x = evaluator_features(result(), probs, top_n=3)
    values = dict(zip(FEATURES, x, strict=True))
    assert values["ce_top1"] == 4.0
    assert values["ce_gap12"] == 2.5
    assert values["bm25_dense_agreement"] == pytest.approx(3 / 10)  # {5, 2, 1}
    assert values["nli_max_entail"] == pytest.approx(0.6)
    assert values["nli_max_contradict"] == pytest.approx(0.8)


def test_gold_in_top() -> None:
    assert gold_in_top(result(), {9}) == 1
    assert gold_in_top(result(), {7}) == 0
    assert gold_in_top(result(), set()) == 0


def test_rrf_rewards_agreement() -> None:
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 3, 4]]) == [2, 3, 1, 4]


def test_stratified_folds_spread_each_class() -> None:
    y = np.array([1] * 10 + [0] * 25)
    folds = stratified_folds(y, 5, seed=0)
    for f in range(5):
        assert (y[folds == f] == 1).sum() == 2
        assert (y[folds == f] == 0).sum() == 5


def _synthetic(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(FEATURES)))
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * x[:, 0] - x[:, 3])))).astype(int)
    return x, y


def test_out_of_fold_scores_are_informative() -> None:
    from crag_verifier.calibration import auroc

    x, y = _synthetic()
    oof = out_of_fold_scores(x, y, k=5, seed=0)
    assert oof.shape == y.shape
    assert auroc(oof, y) > 0.75


@pytest.mark.parametrize("method", ["raw", "platt", "isotonic"])
def test_evaluator_round_trip(method: str) -> None:
    x, y = _synthetic()
    ev, p_oof = RetrievalEvaluator.fit(x, y, method)
    assert p_oof.min() >= 0 and p_oof.max() <= 1
    clone = RetrievalEvaluator.from_dict(ev.to_dict())
    np.testing.assert_allclose(ev.confidence(x), clone.confidence(x))


def test_evaluator_rejects_policies_fitted_on_other_features() -> None:
    x, y = _synthetic()
    ev, _ = RetrievalEvaluator.fit(x, y, "platt")
    d = ev.to_dict()
    d["features"] = ["ce_top1"]
    with pytest.raises(ValueError):
        RetrievalEvaluator.from_dict(d)


def test_select_calibration_reports_both_losses() -> None:
    x, y = _synthetic()
    method, losses = select_calibration(out_of_fold_scores(x, y), y)
    assert method in losses and set(losses) == {"platt", "isotonic"}
    assert losses[method] == min(losses.values())
