# 4. The official evaluator scores every run; the in-repo metric must agree with it

Status: accepted

## Context

Claim-verification numbers are only comparable with published SciFact results if they
come from the official scoring code. Bootstrap confidence intervals, however, need the
metric recomputed on a thousand resamples of the dev claims, which is slow and awkward
through a script that reads and writes files.

## Decision

* `crag-verifier fetch` downloads `evaluator/eval.py` from `allenai/scifact-evaluator`
  (Apache-2.0) at commit `66feffc5`, verified by sha256. Every benchmark run writes each
  system's predictions to JSONL and runs that script on them with `python -X utf8`, so the
  script's default-encoding `open` behaves the same on Windows and Linux.
* `scifact_metrics.py` re-implements the same counting per claim. The benchmark checks that
  the two agree to 1e-9 for every system (`metric_parity_with_official` in the results
  JSON), and a committed fixture of edge cases holds the official script's recorded output
  for the offline test suite.
* Paired bootstrap: every system is scored on the same resampled claim indices, so the
  confidence interval of the difference to standard RAG is a paired one.

## Consequences

* Headline F1 values in the README are the official script's output.
* If the official semantics ever change, the parity check fails loudly instead of the two
  implementations drifting apart silently.

## Alternatives rejected

* **Only the in-repo metric.** Not credible as an external reference.
* **Calling the official script once per bootstrap resample.** Thousands of subprocesses
  for no extra fidelity once parity is enforced.
