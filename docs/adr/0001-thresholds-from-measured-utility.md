# 1. Three-way thresholds come from measured utilities, not from fixed cut-offs

Status: accepted

## Context

CRAG triages each query into Correct, Ambiguous or Incorrect by comparing the retrieval
evaluator's score with two thresholds. The original paper sets them by hand per dataset.
Here the actions are concrete and have different consequences for the SciFact metric:

* CORRECT keeps the top-3 abstracts and refines them into strips;
* AMBIGUOUS rewrites the claim into keyphrases, re-retrieves densely, merges and re-ranks;
* INCORRECT abstains (NOINFO), which can produce neither a false positive nor a true one.

Hand-set cut-offs would hide the question this project wants to answer: when is it worth
acting on retrieval at all?

## Decision

1. The evaluator predicts `p = P(a gold abstract is in the top 3)`. Its logistic scorer is
   trained on SciFact train; the calibrator (Platt or isotonic, chosen by held-out
   log-loss) is fitted on 5-fold out-of-fold scores, so it never sees a score produced by
   a model trained on the same claim. The train probabilities that the fit below uses are
   nested: for each outer fold, scorer and calibrator are refitted on the other folds.
2. Every train claim is run through both acting branches. Each outcome is scored with a
   per-claim utility: correctly labelled abstracts minus wrongly labelled ones (abstaining
   scores 0). The labels come from an aggregator fitted on the other folds, so a claim
   never grades itself. Averaging by whether retrieval was correct gives a 3x2 utility
   table.
3. Expected utility of each action is then linear in `p`. The thresholds are the switch
   points of the upper envelope of the three lines (`policy.envelope`). Exact ties go to
   the cheaper action (abstain, then refine, then re-retrieve). When AMBIGUOUS is not on
   the envelope, `lower == upper` and the single cut point goes to INCORRECT; a cut at 0
   means "never abstain", so `p = 0` then goes to CORRECT. No confidence can reach an
   action the envelope does not contain.
4. If the measured envelope does not visit the actions in INCORRECT, AMBIGUOUS, CORRECT
   order, no two-threshold rule can follow it; the fit then picks the threshold pair with
   the highest expected utility over the held-out train probabilities (`method: "grid"`
   in the fit report).

Nothing on SciFact dev is used to choose thresholds, calibration or aggregator settings.

## Consequences

* Calibration is load-bearing: a miscalibrated `p` moves claims across the switch points.
  This is why ECE is reported next to AUROC.
* If re-retrieval never beats refinement on train, AMBIGUOUS collapses and the policy is
  honestly two-way (`lower == upper`). The benchmark reports the action mix as measured.
* The utility is a proxy for F1, not F1 itself. It is linear per claim, which is what makes
  the expected-utility argument work; the false-positive cost is a parameter (`fp_cost`,
  default 1).

## Alternatives rejected

* **Fixed thresholds (for example 0.3 and 0.7).** Arbitrary, and they ignore what each
  action actually earns on this data.
* **Direct grid search of the thresholds against realised train utility.** Invariant to any
  monotone transform of the score, so calibration would not matter, and with about 800
  claims the realised-utility surface is a noisy step function.
* **Optimising dev F1.** Leaks the evaluation set into the policy.

## Outcome on SciFact train

The published fit (`train_fit.thresholds` in `bench/results/scifact-dev.json`) did not
need the fallback. With strip-refined evidence, re-retrieving earned one label less than
refining over the 448 claims with a correct retrieval and lost one label more over the 361
with a wrong one. AMBIGUOUS is therefore dominated at every `p`, and the envelope is
abstain, then refine, switching at `p` = 0.68 (lower = upper). With sentence-level evidence,
acting lost labels even on correct retrievals, so the envelope is "abstain" everywhere
(lower = upper = 1.0).

An earlier fit scored the utilities with an in-sample aggregator fitted on a coarser
threshold grid. It put refining and re-retrieving within a few thousandths of each other,
its envelope was non-monotone, and the grid fallback chose "abstain or re-retrieve"
(lower 0.55, upper 1.0). Both outcomes come from the measured utilities rather than
tuning. In both, the three-way design collapsed to two ways, and the README reports that
as a result.
