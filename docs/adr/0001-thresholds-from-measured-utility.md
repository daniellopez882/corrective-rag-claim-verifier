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
   a model trained on the same claim.
2. Every train claim is run through both acting branches. Each outcome is scored with a
   per-claim utility: correctly labelled abstracts minus wrongly labelled ones (abstaining
   scores 0). Averaging by whether retrieval was correct gives a 3x2 utility table.
3. Expected utility of each action is then linear in `p`. The thresholds are the switch
   points of the upper envelope of the three lines (`policy.envelope`). Exact ties go to
   the cheaper action (abstain, then refine, then re-retrieve).
4. If the measured envelope does not visit the actions in INCORRECT, AMBIGUOUS, CORRECT
   order, no two-threshold rule can follow it; the fit then picks the threshold pair with
   the highest expected utility over the out-of-fold train probabilities (`method: "grid"`
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
