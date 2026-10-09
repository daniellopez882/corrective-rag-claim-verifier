from __future__ import annotations

import itertools

import numpy as np
import pytest

from crag_verifier.aggregate import (
    GRID,
    AggregatorParams,
    DocEvidence,
    fit_aggregator,
    label_doc,
    sentence_evidence,
    strip_evidence,
    verdict,
)
from crag_verifier.data import Claim, GoldEvidence
from crag_verifier.fit import predictions_from
from crag_verifier.scifact_metrics import evaluate


def test_sentence_evidence_takes_max_and_ranked_rationale() -> None:
    probs = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.55, 0.4, 0.05],
            [0.2, 0.1, 0.7],
            [0.3, 0.6, 0.1],
        ]
    )
    ev = sentence_evidence(5, probs)
    assert ev.entail == pytest.approx(0.7) and ev.contradict == pytest.approx(0.7)
    assert ev.support_rationale == (1, 2)  # best, then others >= 0.5
    assert ev.contradict_rationale == (3,)


def test_strip_evidence_without_kept_strips_reads_nothing() -> None:
    assert strip_evidence(1, None, []) == DocEvidence(1, 0.0, 0.0, (), ())
    ev = strip_evidence(1, np.array([0.6, 0.3, 0.1]), [4, 2])
    assert ev.support_rationale == (2, 4) and ev.entail == pytest.approx(0.6)


@pytest.mark.parametrize(
    "entail,contradict,expected",
    [(0.9, 0.1, "SUPPORT"), (0.1, 0.9, "CONTRADICT"), (0.4, 0.4, None), (0.6, 0.6, "SUPPORT")],
)
def test_label_doc(entail: float, contradict: float, expected: str | None) -> None:
    ev = DocEvidence(1, entail, contradict, (0,), (1,))
    decision = label_doc(ev, AggregatorParams(0.5, 0.5))
    assert decision.label == expected
    assert decision.rationale == ({"SUPPORT": (0,), "CONTRADICT": (1,)}.get(expected or "", ()))


def test_verdict_picks_the_most_confident_labelled_abstract() -> None:
    params = AggregatorParams(0.5, 0.5)
    decisions = [
        label_doc(DocEvidence(1, 0.6, 0.0, (0,), (0,)), params),
        label_doc(DocEvidence(2, 0.0, 0.9, (1,), (1,)), params),
        label_doc(DocEvidence(3, 0.1, 0.1, (1,), (1,)), params),
    ]
    assert verdict(decisions) == ("CONTRADICT", pytest.approx(0.9))
    assert verdict(decisions[2:]) == ("NOINFO", 0.0)


def test_fit_aggregator_finds_the_grid_optimum() -> None:
    rng = np.random.default_rng(4)
    claims, evidence = [], []
    for cid in range(25):
        label = ["SUPPORT", "CONTRADICT"][cid % 2]
        gold = {100 + cid: GoldEvidence(label, ((0,),))} if cid % 5 else {}
        claims.append(Claim(cid, f"claim {cid}", gold))
        docs = []
        for d in (100 + cid, 900 + cid):
            e, c = rng.random(), rng.random()
            docs.append(DocEvidence(d, float(e), float(c), (0,), (0,)))
        evidence.append(docs)
    params, f1 = fit_aggregator(claims, evidence)
    best = 0.0
    for ts, tc in itertools.product(GRID, GRID):
        p = AggregatorParams(float(ts), float(tc))
        preds = [predictions_from(c.id, ev, p) for c, ev in zip(claims, evidence, strict=True)]
        best = max(best, evaluate(claims, preds)["abstract_label_only_f1"])
    assert f1 == pytest.approx(best)
    preds = [predictions_from(c.id, ev, params) for c, ev in zip(claims, evidence, strict=True)]
    assert evaluate(claims, preds)["abstract_label_only_f1"] == pytest.approx(f1)
