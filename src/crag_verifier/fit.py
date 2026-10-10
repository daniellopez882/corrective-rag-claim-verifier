"""Fit every learned piece of the policy on SciFact train, and nothing on dev.

Order matters because each step consumes the previous one:

1. retrieve and featurise every train claim; y = "a gold abstract is in the top 3";
2. strip relevance threshold (sentence-selection F1 against gold rationales);
3. NLI aggregator thresholds per evidence mode (abstract label-only F1);
4. run the CORRECT and AMBIGUOUS actions on every claim and score each outcome with the
   per-claim utility TP - fp_cost * FP (abstaining scores 0). Each claim is labelled by
   an aggregator fitted on the other folds, so the utilities are held-out estimates;
5. evaluator: logistic scorer, cross-validated calibration (Platt or isotonic, chosen on
   held-out log-loss), and nested held-out probabilities;
6. three-way thresholds per evidence mode from the utility table.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from crag_verifier.aggregate import (
    AggregatorParams,
    DocEvidence,
    EvidenceMode,
    fit_aggregator,
    label_doc,
)
from crag_verifier.calibration import auroc, brier_score, expected_calibration_error
from crag_verifier.data import Claim
from crag_verifier.evaluator import (
    RetrievalEvaluator,
    gold_in_top,
    out_of_fold_scores,
    select_calibration,
    stratified_folds,
)
from crag_verifier.pipeline import FittedPolicy, InitialAnalysis, Pipeline
from crag_verifier.policy import (
    ACTIONS,
    Action,
    UtilityTable,
    action_counts,
    decide,
    envelope,
    fit_thresholds,
)
from crag_verifier.retrieval import RetrievalResult
from crag_verifier.scifact_metrics import claim_counts, evaluate
from crag_verifier.strips import rationale_ids, select

Progress = Callable[[str, int, int], None]


class StderrProgress:
    """Prints a line every ``every`` claims with the time spent in the current stage."""

    def __init__(self, every: int = 25):
        self.every = every
        self.started: dict[str, float] = {}

    def __call__(self, stage: str, done: int, total: int) -> None:
        start = self.started.setdefault(stage, time.perf_counter())
        if done == total or (done and done % self.every == 0):
            elapsed = time.perf_counter() - start
            print(f"[{stage}] {done}/{total} claims, {elapsed:.0f}s", file=sys.stderr, flush=True)


def _each[T](items: Sequence[T], stage: str, progress: Progress | None) -> Iterator[T]:
    for i, item in enumerate(items):
        if progress:
            progress(stage, i, len(items))
        yield item
    if progress:
        progress(stage, len(items), len(items))


@dataclass
class ClaimRun:
    """Everything computed for one claim that does not depend on fitted parameters."""

    claim: Claim
    initial: InitialAnalysis
    y: int
    rewritten: str
    expanded: RetrievalResult


def run_claims(
    pipeline: Pipeline, claims: Sequence[Claim], stage: str, progress: Progress | None
) -> list[ClaimRun]:
    runs = []
    for claim in _each(claims, stage, progress):
        initial = pipeline.analyse(claim.text)
        rewritten, expanded = pipeline.expand(claim.text, initial.retrieval)
        y = gold_in_top(initial.retrieval, set(claim.evidence), pipeline.top_n)
        runs.append(ClaimRun(claim, initial, y, rewritten, expanded))
    return runs


def predictions_from(
    claim_id: int, evidence: Sequence[DocEvidence], params: AggregatorParams
) -> dict[str, Any]:
    decisions = [label_doc(e, params) for e in evidence]
    return {
        "id": claim_id,
        "evidence": {
            str(d.doc_id): {"sentences": list(d.rationale), "label": d.label}
            for d in decisions
            if d.label is not None
        },
    }


def claim_utility(claim: Claim, prediction: dict[str, Any], fp_cost: float) -> float:
    counts = claim_counts(claim, prediction)
    tp = counts[2]  # abstract_correct_label_only
    fp = counts[1] - tp  # abstract_retrieved - correct
    return float(tp - fp_cost * fp)


def fit_strip_threshold(
    pipeline: Pipeline, runs: Sequence[ClaimRun], progress: Progress | None = None
) -> tuple[float, float]:
    """Relevance cut-off that maximises sentence-selection F1 over the top-3 abstracts."""
    relevance = [
        pipeline.strip_relevance(r.claim.text, r.initial.retrieval.top(pipeline.top_n))
        for r in _each(runs, "fit:strip-relevance", progress)
    ]
    all_scores = np.concatenate([s for rel in relevance for _, s in rel.values()] or [np.zeros(1)])
    candidates = np.unique(np.round(np.percentile(all_scores, np.arange(0, 100.1, 2.5)), 4))
    best = (-1.0, float(candidates[0]))
    claims = [r.claim for r in runs]
    for t in candidates:
        preds = []
        for r, rel in zip(runs, relevance, strict=True):
            evidence = {}
            for doc_id, (strips, scores) in rel.items():
                kept = select(strips, scores, float(t))
                if kept:
                    # The label does not enter sentence-selection scoring.
                    evidence[str(doc_id)] = {"sentences": rationale_ids(kept), "label": "SUPPORT"}
            preds.append({"id": r.claim.id, "evidence": evidence})
        f1 = evaluate(claims, preds)["sentence_selection_f1"]
        if f1 > best[0] + 1e-12:
            best = (f1, float(t))
    return best[1], best[0]


def _calibration_report(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "ece_10bins": expected_calibration_error(p, y),
        "brier": brier_score(p, y),
        "auroc": auroc(p, y),
    }


def fit_policy(
    pipeline: Pipeline,
    claims: Sequence[Claim],
    *,
    calibration: str = "auto",
    folds: int = 5,
    seed: int = 0,
    fp_cost: float = 1.0,
    progress: Progress | None = None,
) -> tuple[FittedPolicy, dict[str, Any], list[ClaimRun]]:
    runs = run_claims(pipeline, claims, "fit:retrieve", progress)
    y = np.array([r.y for r in runs], dtype=np.int64)
    x = np.stack([r.initial.features for r in runs])

    min_relevance, strip_f1 = fit_strip_threshold(pipeline, runs, progress)

    top = pipeline.top_n
    aggregator: dict[EvidenceMode, AggregatorParams] = {}
    aggregator_report: dict[str, Any] = {}
    evidence_correct: dict[EvidenceMode, list[list[DocEvidence]]] = {}
    evidence_ambiguous: dict[EvidenceMode, list[list[DocEvidence]]] = {}
    for mode in EvidenceMode:
        evidence_correct[mode] = [
            pipeline.evidence(
                r.claim.text,
                r.initial.retrieval.top(top),
                mode,
                min_relevance,
                r.initial.sentence_probs,
            )
            for r in _each(runs, f"fit:evidence-{mode.value}", progress)
        ]
        fitted = fit_aggregator(claims, evidence_correct[mode])
        aggregator[mode] = fitted.params
        aggregator_report[mode.value] = fitted.report()
        evidence_ambiguous[mode] = [
            pipeline.evidence(
                r.claim.text, r.expanded.top(top), mode, min_relevance, r.initial.sentence_probs
            )
            for r in _each(runs, f"fit:expanded-evidence-{mode.value}", progress)
        ]

    # Utilities are scored with an aggregator that did not see the claim, as on dev.
    k_folds = max(2, min(folds, int(y.sum()), int(len(y) - y.sum())))
    fold_of = stratified_folds(y, k_folds, seed)
    utilities: dict[EvidenceMode, dict[Action, np.ndarray]] = {}
    for mode in EvidenceMode:
        u_correct = np.empty(len(runs))
        u_ambiguous = np.empty(len(runs))
        for f in range(k_folds):
            train_idx = np.flatnonzero(fold_of != f)
            params = fit_aggregator(
                [claims[i] for i in train_idx], [evidence_correct[mode][i] for i in train_idx]
            ).params
            for i in np.flatnonzero(fold_of == f):
                r = runs[i]
                for out, ev in (
                    (u_correct, evidence_correct[mode][i]),
                    (u_ambiguous, evidence_ambiguous[mode][i]),
                ):
                    out[i] = claim_utility(
                        r.claim, predictions_from(r.claim.id, ev, params), fp_cost
                    )
        utilities[mode] = {
            Action.INCORRECT: np.zeros(len(runs)),
            Action.AMBIGUOUS: u_ambiguous,
            Action.CORRECT: u_correct,
        }

    oof_raw = out_of_fold_scores(x, y, k=folds, seed=seed)
    selection_losses: dict[str, float] = {}
    method = calibration
    if calibration == "auto":
        method, selection_losses = select_calibration(oof_raw, y)
    evaluator, p_held_out = RetrievalEvaluator.fit(x, y, method, k=folds, seed=seed)

    thresholds = {}
    threshold_report: dict[str, Any] = {}
    for mode in EvidenceMode:
        table = UtilityTable.from_outcomes(utilities[mode], y)
        thresholds[mode], how = fit_thresholds(table, p_held_out)
        chosen = [decide(float(p), thresholds[mode]) for p in p_held_out]
        realised = float(np.mean([utilities[mode][a][i] for i, a in enumerate(chosen)]))
        threshold_report[mode.value] = {
            "utility_table": table.to_dict(),
            "envelope": [
                {"from": lo, "to": hi, "action": a.value} for lo, hi, a in envelope(table)
            ],
            "thresholds": thresholds[mode].to_dict(),
            "method": how,
            "held_out_mean_utility_policy": realised,
            "mean_utility_always": {a.value: float(np.mean(utilities[mode][a])) for a in ACTIONS},
            "held_out_action_counts": action_counts(chosen),
        }

    report = {
        "n_claims": len(runs),
        "retrieval_correct_rate": float(y.mean()),
        "strip_min_relevance": min_relevance,
        "strip_sentence_selection_f1": strip_f1,
        "aggregator": aggregator_report,
        "aggregator_note": "label_only_f1 is in-sample on train; the utility table below is "
        "scored with aggregators fitted on the other folds",
        "calibration_method": method,
        "calibration_selection_logloss": selection_losses,
        "held_out_calibrated": _calibration_report(p_held_out, y),
        "fp_cost": fp_cost,
        "folds": folds,
        "thresholds": threshold_report,
    }
    policy = FittedPolicy(
        evaluator=evaluator,
        thresholds=thresholds,
        aggregator=aggregator,
        strip_min_relevance=min_relevance,
        provenance={
            "fitted_on": f"{len(runs)} claims",
            "calibration": method,
            "fp_cost": fp_cost,
            "folds": folds,
            "seed": seed,
        },
    )
    return policy, report, runs
