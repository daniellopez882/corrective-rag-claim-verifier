from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from crag_verifier.calibration import (
    IsotonicCalibrator,
    LogisticModel,
    PlattCalibrator,
    auroc,
    brier_score,
    calibrator_from_dict,
    expected_calibration_error,
    fit_logistic,
    log_loss,
    reliability_bins,
)

labelled = st.integers(min_value=6, max_value=80).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(-5, 5).map(float), min_size=n, max_size=n),
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
    )
)


@settings(max_examples=80, deadline=None)
@given(data=labelled)
def test_auroc_matches_sklearn_including_ties(data: tuple[list[float], list[int]]) -> None:
    scores, y = np.array(data[0]), np.array(data[1])
    assume(0 < y.sum() < len(y))
    assert auroc(scores, y) == pytest.approx(roc_auc_score(y, scores), abs=1e-12)


def test_auroc_is_undefined_for_one_class() -> None:
    assert np.isnan(auroc(np.array([0.1, 0.9]), np.array([1, 1])))


def test_ece_hand_computed_reference() -> None:
    # Bin [0.1, 0.2): confidences 0.1, 0.1, outcomes 0, 1 -> |0.5 - 0.1| * 2
    # Bin [0.9, 1.0]: confidences 0.9, 1.0, outcomes 1, 1 -> |1.0 - 0.95| * 2
    p = np.array([0.1, 0.1, 0.9, 1.0])
    y = np.array([0, 1, 1, 1])
    assert expected_calibration_error(p, y) == pytest.approx((0.4 * 2 + 0.05 * 2) / 4)


@settings(max_examples=60, deadline=None)
@given(
    p=st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=60),
    seed=st.integers(0, 1000),
)
def test_ece_matches_a_brute_force_loop(p: list[float], seed: int) -> None:
    probs = np.array(p)
    y = np.random.default_rng(seed).integers(0, 2, size=len(probs))
    total = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        members = [i for i, v in enumerate(probs) if lo <= v < hi or (b == 9 and v == 1.0)]
        if members:
            total += len(members) * abs(y[members].mean() - probs[members].mean())
    ece = expected_calibration_error(probs, y)
    assert ece == pytest.approx(total / len(probs), abs=1e-12)
    assert 0.0 <= ece <= 1.0


def test_reliability_bins_report_empty_bins_as_none() -> None:
    bins = reliability_bins(np.array([0.05, 0.95]), np.array([0, 1]))
    assert len(bins) == 10
    assert bins[0]["count"] == 1 and bins[5]["mean_confidence"] is None


@settings(max_examples=60, deadline=None)
@given(data=labelled, probe=st.lists(st.floats(-8, 8, allow_nan=False), min_size=1, max_size=20))
def test_isotonic_matches_sklearn_and_is_monotone(
    data: tuple[list[float], list[int]], probe: list[float]
) -> None:
    scores, y = np.array(data[0]), np.array(data[1], dtype=float)
    ours = IsotonicCalibrator.fit(scores, y)
    ref = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(scores, y)
    grid = np.sort(np.array(probe))
    np.testing.assert_allclose(ours.predict(grid), ref.predict(grid), atol=1e-12)
    assert np.all(np.diff(ours.predict(grid)) >= -1e-12)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), slope=st.floats(0.3, 3.0))
def test_platt_is_monotone_increasing_for_informative_scores(seed: int, slope: float) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=200)
    y = (rng.random(200) < 1 / (1 + np.exp(-slope * scores))).astype(float)
    assume(0 < y.sum() < 200)
    cal = PlattCalibrator.fit(scores, y)
    grid = np.linspace(-3, 3, 25)
    assert cal.a > 0
    assert np.all(np.diff(cal.predict(grid)) > 0)


def test_platt_matches_sklearns_sigmoid_calibration() -> None:
    calibration = pytest.importorskip("sklearn.calibration")
    sigmoid_fit = getattr(calibration, "_sigmoid_calibration", None)
    if sigmoid_fit is None:
        pytest.skip("this scikit-learn has no _sigmoid_calibration")
    rng = np.random.default_rng(3)
    scores = rng.normal(size=300)
    y = (rng.random(300) < 1 / (1 + np.exp(-1.7 * scores + 0.4))).astype(float)
    a_ref, b_ref = sigmoid_fit(scores, y)  # sklearn: p = 1 / (1 + exp(a * s + b))
    cal = PlattCalibrator.fit(scores, y)
    assert cal.a == pytest.approx(-a_ref, rel=1e-4)
    assert cal.b == pytest.approx(-b_ref, rel=1e-4, abs=1e-6)


def test_logistic_matches_sklearn_l2() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    y = (rng.random(300) < 1 / (1 + np.exp(-(x @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.3)))).astype(
        int
    )
    coef, intercept = fit_logistic(x, y, l2=1.0)
    ref = LogisticRegression(C=1.0, tol=1e-10, max_iter=10_000).fit(x, y)
    np.testing.assert_allclose(coef, ref.coef_[0], rtol=1e-5, atol=1e-6)
    assert intercept == pytest.approx(ref.intercept_[0], rel=1e-5, abs=1e-6)


def test_logistic_model_round_trips_and_handles_constant_features() -> None:
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(size=50), np.ones(50)])
    y = (x[:, 0] > 0).astype(int)
    model = LogisticModel.fit(x, y)
    clone = LogisticModel.from_dict(model.to_dict())
    np.testing.assert_allclose(model.predict_proba(x), clone.predict_proba(x))
    assert np.isfinite(model.decision_function(x)).all()


def test_calibrators_round_trip_through_dicts() -> None:
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = np.array([0, 0, 1, 0, 1, 1])
    for cal in (PlattCalibrator.fit(scores, y), IsotonicCalibrator.fit(scores, y)):
        clone = calibrator_from_dict(cal.to_dict())
        np.testing.assert_allclose(cal.predict(scores), clone.predict(scores))


def test_scoring_rules() -> None:
    p = np.array([0.0, 1.0, 0.5])
    y = np.array([0, 1, 1])
    assert brier_score(p, y) == pytest.approx(0.25 / 3)
    assert log_loss(p, y) == pytest.approx(-np.log(0.5) / 3, rel=1e-6)
