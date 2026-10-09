# 2. A three-class NLI model replaces the binary xtremedistil checkpoint

Status: accepted

## Context

The plan named `xtremedistil-l6-h256` as the NLI model. The public checkpoints of that
architecture fine-tuned for NLI (`MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary`
and `MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33`) are binary: entailment versus
not-entailment. This project needs a contradiction probability in three places: the
evaluator feature `nli_max_contradict`, the CONTRADICT label of the aggregator, and the
rationale of contradicting abstracts. More than a third of the SciFact dev rationale sets
are CONTRADICT, so dropping the label is not an option.

## Decision

Use `cross-encoder/nli-MiniLM2-L6-H768` (Apache-2.0; contradiction, entailment, neutral),
loading the `onnx/model_quint8_avx2.onnx` export published in the same repository, pinned
to commit `b95119ce`. The label order is read from the model's `config.json`, never
assumed.

The other two models follow the same rule: official ONNX exports at pinned commits
(`BAAI/bge-small-en-v1.5` fp32, `cross-encoder/ms-marco-MiniLM-L6-v2` uint8).

## Consequences

* The NLI model costs roughly four times the per-token compute of xtremedistil-l6-h256
  (hidden size 768 instead of 256). With the cross-encoder rerank it dominates both the
  benchmark runtime and `/verify` latency.
* It is a general-domain MNLI/SNLI model used zero-shot on biomedical claims. The
  aggregator thresholds are fitted on SciFact train to move its operating point, but no
  weights are fine-tuned.
* Quantised weights trade some accuracy for speed; the integration tests check that the
  quantised model still separates obvious entailment and contradiction pairs.

## Alternatives rejected

* **Binary xtremedistil plus a negated hypothesis to fake contradiction.** An untested
  heuristic with unknown behaviour on scientific negation.
* **`cross-encoder/nli-deberta-v3-xsmall`.** Similar accuracy, but twelve layers and
  DeBERTa's relative attention make it slower in ONNX Runtime on this CPU.
* **`MoritzLaurer/MiniLM-L6-mnli` (hidden size 384).** Faster, but it ships PyTorch weights
  only and its model card states no licence.
