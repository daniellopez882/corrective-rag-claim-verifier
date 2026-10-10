"""NLI label aggregation per abstract, and the claim-level verdict.

Two evidence modes produce the same ``DocEvidence`` record:

* ``sentences``: the NLI model reads every sentence on its own; the abstract's scores are
  the maximum entailment and contradiction over its sentences, and the rationale is the
  best sentence plus up to two more whose probability for that label is at least 0.5.
* ``strips``: the refined passage (kept strips, recomposed in order) is read once; the
  rationale is the kept strips.

The label thresholds are fitted on SciFact train by ``fit_aggregator``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from crag_verifier.data import Claim, Label
from crag_verifier.models import CONTRADICT, ENTAIL
from crag_verifier.strips import MAX_RATIONALE_SENTENCES

SECONDARY_RATIONALE_MIN = 0.5

Verdict = Literal["SUPPORT", "CONTRADICT", "NOINFO"]


class EvidenceMode(StrEnum):
    SENTENCES = "sentences"
    STRIPS = "strips"


@dataclass(frozen=True)
class DocEvidence:
    doc_id: int
    entail: float
    contradict: float
    support_rationale: tuple[int, ...]
    contradict_rationale: tuple[int, ...]


@dataclass(frozen=True)
class AggregatorParams:
    support: float  # minimum entailment probability for SUPPORT
    contradict: float  # minimum contradiction probability for CONTRADICT

    def to_dict(self) -> dict[str, float]:
        return {"support": self.support, "contradict": self.contradict}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AggregatorParams:
        return cls(float(d["support"]), float(d["contradict"]))


@dataclass(frozen=True)
class DocDecision:
    doc_id: int
    label: Label | None  # None is NOINFO for this abstract
    probability: float
    rationale: tuple[int, ...]


def _ranked_rationale(sentence_ids: Sequence[int], probs: np.ndarray) -> tuple[int, ...]:
    order = np.argsort(-probs, kind="stable")
    keep = [int(order[0])] + [
        int(i) for i in order[1:MAX_RATIONALE_SENTENCES] if probs[i] >= SECONDARY_RATIONALE_MIN
    ]
    return tuple(sentence_ids[i] for i in keep)


def sentence_evidence(doc_id: int, probs: np.ndarray) -> DocEvidence:
    """Evidence from per-sentence NLI; ``probs`` row i belongs to sentence i."""
    if len(probs) == 0:
        return DocEvidence(doc_id, 0.0, 0.0, (), ())
    ids = list(range(len(probs)))
    return DocEvidence(
        doc_id,
        float(probs[:, ENTAIL].max()),
        float(probs[:, CONTRADICT].max()),
        _ranked_rationale(ids, probs[:, ENTAIL]),
        _ranked_rationale(ids, probs[:, CONTRADICT]),
    )


def strip_evidence(
    doc_id: int, passage_probs: np.ndarray | None, kept: Sequence[int]
) -> DocEvidence:
    """Evidence from the recomposed passage; no kept strips means nothing to read."""
    if passage_probs is None or not kept:
        return DocEvidence(doc_id, 0.0, 0.0, (), ())
    rationale = tuple(sorted(kept))
    return DocEvidence(
        doc_id,
        float(passage_probs[ENTAIL]),
        float(passage_probs[CONTRADICT]),
        rationale,
        rationale,
    )


def label_doc(ev: DocEvidence, params: AggregatorParams) -> DocDecision:
    options: list[tuple[float, Label, tuple[int, ...]]] = []
    if ev.entail >= params.support and ev.support_rationale:
        options.append((ev.entail, "SUPPORT", ev.support_rationale))
    if ev.contradict >= params.contradict and ev.contradict_rationale:
        options.append((ev.contradict, "CONTRADICT", ev.contradict_rationale))
    if not options:
        return DocDecision(ev.doc_id, None, max(ev.entail, ev.contradict), ())
    prob, label, rationale = max(options, key=lambda o: o[0])
    return DocDecision(ev.doc_id, label, prob, rationale)


def verdict(decisions: Sequence[DocDecision]) -> tuple[Verdict, float]:
    """Claim-level verdict: the most confident labelled abstract, else NOINFO."""
    best: DocDecision | None = None
    for d in decisions:
        if d.label is not None and (best is None or d.probability > best.probability):
            best = d
    if best is None or best.label is None:
        return "NOINFO", 0.0
    return best.label, best.probability


GRID = np.round(np.arange(0.0, 1.0, 0.05), 2)


def threshold_candidates(values: np.ndarray) -> np.ndarray:
    """The 0.05 grid from 0 plus the 2.5% quantiles of the observed probabilities.

    The quantiles let the fit settle between grid steps where the probabilities bunch up,
    which the strip-mode entailment does close to zero.
    """
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 41)) if len(values) else np.empty(0)
    return np.unique(np.concatenate([GRID, np.round(quantiles, 4)]))


@dataclass(frozen=True)
class AggregatorFit:
    params: AggregatorParams
    label_only_f1: float  # on the claims it was fitted on
    edges: dict[str, str]  # threshold name -> "lowest" or "highest" candidate it landed on

    def report(self) -> dict[str, Any]:
        return {**self.params.to_dict(), "label_only_f1": self.label_only_f1, "edges": self.edges}


def fit_aggregator(
    claims: Sequence[Claim], evidence: Sequence[Sequence[DocEvidence]]
) -> AggregatorFit:
    """Search both thresholds for the best abstract label-only F1 on these claims."""
    gold_total = sum(len(c.evidence) for c in claims)
    rows = []
    for claim, docs in zip(claims, evidence, strict=True):
        for ev in docs:
            gold = claim.evidence.get(ev.doc_id)
            code = 0 if gold is None else (1 if gold.label == "SUPPORT" else 2)
            rows.append(
                (
                    ev.entail,
                    ev.contradict,
                    code,
                    bool(ev.support_rationale),
                    bool(ev.contradict_rationale),
                )
            )
    if not rows:
        raise ValueError("no retrieved abstracts to fit the aggregator on")
    arr = np.array(rows, dtype=np.float64)
    entail, contradict, code = arr[:, 0], arr[:, 1], arr[:, 2]
    has_sup, has_con = arr[:, 3] > 0, arr[:, 4] > 0
    support_grid = threshold_candidates(entail[has_sup])
    contradict_grid = threshold_candidates(contradict[has_con])
    best: tuple[float, float, float] | None = None
    for ts in support_grid:
        sup_ok = (entail >= ts) & has_sup
        for tc in contradict_grid:
            con_ok = (contradict >= tc) & has_con
            pick_sup = sup_ok & (~con_ok | (entail >= contradict))
            pick_con = con_ok & ~pick_sup
            retrieved = int(pick_sup.sum() + pick_con.sum())
            correct = int((pick_sup & (code == 1)).sum() + (pick_con & (code == 2)).sum())
            p = correct / retrieved if retrieved else 0.0
            r = correct / gold_total if gold_total else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            if best is None or f1 > best[0] + 1e-12:
                best = (f1, float(ts), float(tc))
    assert best is not None
    edges = {}
    for name, value, grid in (
        ("support", best[1], support_grid),
        ("contradict", best[2], contradict_grid),
    ):
        if value == grid[0]:
            edges[name] = "lowest"
        elif value == grid[-1]:
            edges[name] = "highest"
    return AggregatorFit(AggregatorParams(best[1], best[2]), best[0], edges)
