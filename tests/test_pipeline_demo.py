"""End to end on the synthetic demo corpus with the lexical model doubles."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from crag_verifier.aggregate import EvidenceMode
from crag_verifier.calibration import PlattCalibrator
from crag_verifier.data import Claim, Corpus
from crag_verifier.pipeline import Explainer, FittedPolicy, Pipeline, System, Verifier
from crag_verifier.policy import Action, Thresholds
from crag_verifier.providers import FakeProvider


def with_thresholds(policy: FittedPolicy, lower: float, upper: float) -> FittedPolicy:
    th = Thresholds(lower, upper)
    return dataclasses.replace(policy, thresholds={m: th for m in EvidenceMode})


def with_confidence(policy: FittedPolicy, p: float) -> FittedPolicy:
    """Same policy, but the evaluator reports confidence p for every claim."""
    constant = PlattCalibrator(a=0.0, b=float(np.log(p / (1 - p))))
    evaluator = dataclasses.replace(policy.evaluator, calibrator=constant)
    return dataclasses.replace(policy, evaluator=evaluator)


def test_policy_round_trips_through_json(demo_policy: FittedPolicy, tmp_path) -> None:
    path = tmp_path / "policy.json"
    demo_policy.save(path)
    loaded = FittedPolicy.load(path)
    assert loaded.to_dict() == demo_policy.to_dict()


def test_correct_action_refines_strips_into_rationales(
    demo_pipeline: Pipeline, demo_policy: FittedPolicy
) -> None:
    policy = with_confidence(with_thresholds(demo_policy, 0.3, 0.7), 0.9)
    result = Verifier(demo_pipeline, policy).verify("Velorin lowers fasting blood glucose in mice.")
    assert result.action is Action.CORRECT
    assert result.verdict == "SUPPORT"
    pred = result.prediction(1)
    assert pred["evidence"]["101"]["label"] == "SUPPORT"
    assert 2 in pred["evidence"]["101"]["sentences"]
    assert [t.step for t in result.trace] == ["retrieve", "nli", "evaluate", "refine", "aggregate"]


def test_contradiction_is_detected(demo_pipeline: Pipeline, demo_policy: FittedPolicy) -> None:
    verifier = Verifier(demo_pipeline, with_confidence(with_thresholds(demo_policy, 0.3, 0.7), 0.9))
    result = verifier.verify("Knockdown of TRX-9 reduces axon growth.", System.CRAG_NO_STRIPS)
    assert result.verdict == "CONTRADICT"
    assert result.prediction(4)["evidence"]["103"]["sentences"][0] == 1


def test_incorrect_action_abstains(demo_pipeline: Pipeline, demo_policy: FittedPolicy) -> None:
    policy = with_confidence(with_thresholds(demo_policy, 0.5, 0.8), 0.2)
    result = Verifier(demo_pipeline, policy).verify("Velorin lowers fasting blood glucose in mice.")
    assert result.action is Action.INCORRECT
    assert result.confidence == pytest.approx(0.2)
    assert result.verdict == "NOINFO" and result.decisions == []
    assert result.prediction(1) == {"id": 1, "evidence": {}}
    assert "abstain" in [t.step for t in result.trace]


def test_ambiguous_action_rewrites_and_re_retrieves(
    demo_pipeline: Pipeline, demo_policy: FittedPolicy
) -> None:
    policy = with_confidence(with_thresholds(demo_policy, 0.3, 0.7), 0.5)
    result = Verifier(demo_pipeline, policy).verify("Strain K4 degrades polyethylene film.")
    assert result.action is Action.AMBIGUOUS
    assert result.rewritten_query and "K4" in result.rewritten_query
    assert "re-retrieve" in [t.step for t in result.trace]
    assert result.verdict == "SUPPORT"


def test_baselines_ignore_the_thresholds(
    demo_pipeline: Pipeline, demo_policy: FittedPolicy
) -> None:
    policy = with_confidence(with_thresholds(demo_policy, 0.5, 0.8), 0.2)
    verifier = Verifier(demo_pipeline, policy)
    claim = "Daily walking improves sleep quality in older adults."
    assert verifier.verify(claim, System.STANDARD_RAG).action is Action.CORRECT
    assert verifier.verify(claim, System.ALWAYS_EXPAND).action is Action.AMBIGUOUS
    assert verifier.verify(claim, System.CRAG_NO_STRIPS).action is Action.INCORRECT


def test_fitted_demo_policy_abstains_on_unrelated_claims_and_acts_on_supported_ones(
    demo_pipeline: Pipeline, demo_policy: FittedPolicy, demo_data: tuple[Corpus, list[Claim]]
) -> None:
    _, claims = demo_data
    verifier = Verifier(demo_pipeline, demo_policy)
    actions = {c.id: verifier.verify(c.text).action for c in claims}
    assert all(actions[c.id] is not Action.INCORRECT for c in claims if c.evidence)
    abstained = {cid for cid, a in actions.items() if a is Action.INCORRECT}
    unrelated = set(range(13, 19))  # share no content word with any abstract
    assert unrelated <= abstained
    assert all(not c.evidence for c in claims if c.id in abstained)


def test_explanation_goes_through_the_provider(
    demo_pipeline: Pipeline, demo_policy: FittedPolicy
) -> None:
    provider = FakeProvider()
    verifier = Verifier(
        demo_pipeline,
        with_confidence(with_thresholds(demo_policy, 0.3, 0.7), 0.9),
        Explainer(provider),
    )
    result = verifier.verify("Velorin lowers fasting blood glucose in mice.", explain=True)
    assert result.explanation and result.explanation.startswith("[fake ")
    _, user = provider.calls[0]
    assert "[101:2] (SUPPORT)" in user
    assert result.trace[-1].step == "explain"
