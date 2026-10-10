from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crag_verifier.policy import (
    ACTIONS,
    Action,
    Thresholds,
    UtilityTable,
    decide,
    envelope,
    fit_thresholds,
)


def table(correct: tuple[float, float, float], wrong: tuple[float, float, float]) -> UtilityTable:
    return UtilityTable(
        dict(zip(ACTIONS, correct, strict=True)), dict(zip(ACTIONS, wrong, strict=True))
    )


def test_boundaries_go_to_the_middle_action() -> None:
    th = Thresholds(0.3, 0.7)
    assert decide(0.2999999, th) is Action.INCORRECT
    assert decide(0.3, th) is Action.AMBIGUOUS
    assert decide(0.5, th) is Action.AMBIGUOUS
    assert decide(0.7, th) is Action.AMBIGUOUS
    assert decide(0.7000001, th) is Action.CORRECT


def test_equal_thresholds_make_a_two_way_rule() -> None:
    th = Thresholds(0.4, 0.4)
    assert decide(0.39, th) is Action.INCORRECT
    assert decide(0.4, th) is Action.INCORRECT  # no AMBIGUOUS band to hedge into
    assert decide(0.41, th) is Action.CORRECT


def test_degenerate_rules_at_the_ends_never_reach_the_absent_actions() -> None:
    always_act = Thresholds(0.0, 0.0)
    assert {decide(p, always_act) for p in (0.0, 0.5, 1.0)} == {Action.CORRECT}
    always_abstain = Thresholds(1.0, 1.0)
    assert {decide(p, always_abstain) for p in (0.0, 0.5, 1.0)} == {Action.INCORRECT}


def test_a_dominated_middle_action_is_not_chosen_at_zero_confidence() -> None:
    # CORRECT beats both other actions everywhere, AMBIGUOUS is heavily penalised.
    t = table(correct=(0.0, -5.0, 1.0), wrong=(0.0, -5.0, 0.5))
    th, how = fit_thresholds(t, np.linspace(0, 1, 11))
    assert how == "envelope"
    assert decide(0.0, th) is Action.CORRECT


@pytest.mark.parametrize("lower,upper", [(-0.1, 0.5), (0.6, 0.5), (0.2, 1.01)])
def test_invalid_thresholds_are_rejected(lower: float, upper: float) -> None:
    with pytest.raises(ValueError):
        Thresholds(lower, upper)


@settings(max_examples=100, deadline=None)
@given(
    a=st.floats(0, 1),
    b=st.floats(0, 1),
    ps=st.lists(st.floats(0, 1), min_size=2, max_size=30),
)
def test_action_never_decreases_as_confidence_rises(a: float, b: float, ps: list[float]) -> None:
    th = Thresholds(min(a, b), max(a, b))
    orders = [decide(p, th).order for p in sorted(ps)]
    assert orders == sorted(orders)


def test_envelope_thresholds_are_the_analytic_crossings() -> None:
    # I: 0 everywhere. A: -0.2 + 0.8p. C: -1 + 2p.
    # A beats I above p = 0.25; C beats A above p = 0.8 / 1.2.
    t = table(correct=(0.0, 0.6, 1.0), wrong=(0.0, -0.2, -1.0))
    th, how = fit_thresholds(t, np.linspace(0, 1, 11))
    assert how == "envelope"
    assert th.lower == pytest.approx(0.25)
    assert th.upper == pytest.approx(0.8 / 1.2)
    assert [s[2] for s in envelope(t)] == [Action.INCORRECT, Action.AMBIGUOUS, Action.CORRECT]


def test_dominated_middle_action_collapses_to_two_way() -> None:
    t = table(correct=(0.0, 0.2, 1.0), wrong=(0.0, -0.5, -1.0))
    th, how = fit_thresholds(t, np.linspace(0, 1, 11))
    assert how == "envelope"
    assert th.lower == th.upper == pytest.approx(0.5)


def test_identical_lines_prefer_the_cheaper_action() -> None:
    t = table(correct=(0.0, 0.5, 0.5), wrong=(0.0, -0.5, -0.5))
    segments = envelope(t)
    assert [s[2] for s in segments] == [Action.INCORRECT, Action.CORRECT]


def test_non_monotone_envelope_falls_back_to_the_best_threshold_rule() -> None:
    # Re-retrieval pays off when retrieval was wrong, so AMBIGUOUS wins at low p,
    # abstaining in the middle, CORRECT at high p: no threshold rule follows that.
    t = table(correct=(0.0, -1.0, 1.0), wrong=(0.0, 0.6, -1.0))
    rng = np.random.default_rng(0)
    p = rng.random(200)
    th, how = fit_thresholds(t, p)
    assert how == "grid"
    eu = t.expected(p)

    def total(th_: Thresholds) -> float:
        return float(sum(eu[i, decide(v, th_).order] for i, v in enumerate(p)))

    grid = np.linspace(0, 1, 101)
    best = max(
        total(Thresholds(lo, hi)) for lo, hi in itertools.combinations_with_replacement(grid, 2)
    )
    assert total(th) == pytest.approx(best)


utility = st.floats(-2.0, 2.0, allow_nan=False)


@settings(max_examples=200, deadline=None)
@given(
    correct=st.tuples(st.just(0.0), utility, utility),
    wrong=st.tuples(st.just(0.0), utility, utility),
)
def test_envelope_rule_is_utility_optimal_at_every_confidence(
    correct: tuple[float, float, float], wrong: tuple[float, float, float]
) -> None:
    t = table(correct, wrong)
    th, how = fit_thresholds(t, np.linspace(0, 1, 11))
    if how != "envelope":
        return  # non-monotone envelopes have no exact threshold rule
    p = np.unique(np.concatenate([np.linspace(0, 1, 201), [th.lower, th.upper]]))
    eu = t.expected(p)
    chosen = eu[np.arange(len(p)), [decide(float(v), th).order for v in p]]
    np.testing.assert_allclose(chosen, eu.max(axis=1), atol=1e-9)


def test_utility_table_needs_both_outcomes() -> None:
    u = {a: np.zeros(3) for a in ACTIONS}
    with pytest.raises(ValueError):
        UtilityTable.from_outcomes(u, np.array([1, 1, 1]))
    tbl = UtilityTable.from_outcomes(
        {
            Action.INCORRECT: np.zeros(4),
            Action.AMBIGUOUS: np.array([1.0, 0.0, -1.0, 0.0]),
            Action.CORRECT: np.array([2.0, 0.0, -1.0, -1.0]),
        },
        np.array([1, 1, 0, 0]),
    )
    assert tbl.correct[Action.CORRECT] == 1.0 and tbl.wrong[Action.AMBIGUOUS] == -0.5
