Hand-written gold claims and predictions covering the evaluator's edge cases (second
rationale, wrong label, abstract outside the gold set, NOINFO claim with a prediction,
gold rationale longer than three sentences, abstention, right label with wrong sentences).

`expected_official_metrics.json` is the output of the pinned official evaluator
(allenai/scifact-evaluator `evaluator/eval.py` at commit 66feffc5) run with `--verbose`
on these two files. `tests/test_scifact_metrics.py` checks the in-repo metric against
it, and re-runs the real script when `crag-verifier fetch` has downloaded it.
