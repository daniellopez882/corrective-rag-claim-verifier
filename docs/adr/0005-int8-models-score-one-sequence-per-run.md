# 5. The int8 models score one sequence per run

Status: accepted

## Context

The cross-encoder and NLI models are the official `model_quint8_avx2.onnx` exports, which
are dynamically quantised: every MatMul input is quantised to uint8 with a scale and zero
point computed at run time from the whole input tensor. With batches of 32 length-sorted,
padded sequences, that tensor holds every row in the batch and its padding, so a pair's
output depended on what else was in the batch. On SciFact pairs the cross-encoder logit of
the same pair moved by up to 0.12, and an NLI probability by up to 0.11, between "scored
alone" and "scored with 19 other abstracts".

That broke two things. The benchmark memoises outputs per (model, pair) in SQLite, so
cached values depended on which run first computed them. In particular, values from smoke
runs on a sampled corpus leaked into the full run. And the service, which scores without
the cache, produced different features from the benchmark for the same claim.

## Decision

Run both int8 models with one sequence per `session.run` (`QUANTISED_BATCH = 1` in
`models.py`). The fp32 dense encoder keeps batching, because padding with an attention
mask changes its output only at float rounding level and the 5,183-abstract index is
expensive to rebuild. The score cache carries a format version (`CACHE_VERSION`), and
opening a cache from an older version drops its stored values.

## Consequences

* A pair's score is a function of the pair alone, so the cache is exact and the service
  reproduces the benchmark's features and decisions for the same claim (checked on claim 13
  of SciFact dev).
* On the i5-8500 it was not slower: on a sample of 100 cross-encoder pairs and 160 NLI
  pairs from SciFact, single-sequence runs finished sooner than padded batches of 32.
* An integration test pins the property: the same pairs scored together and one by one
  must give identical outputs.
* The first full benchmark run, made before this change, was discarded.

## Alternatives rejected

* **Keep batching and key the cache on the whole batch.** Exact, but nearly every lookup
  would miss, and the service would still disagree with the benchmark.
* **Use the fp32 exports.** Batch-independent, but larger downloads and, on CPU, usually
  slower than int8 (not measured here); the int8 exports already pass the checks once
  batching is out of the way.
* **Static quantisation with calibrated scales.** Would need a calibration set and a
  re-export; out of scope for a project that pins official artifacts.
