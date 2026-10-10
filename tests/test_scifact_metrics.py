from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from crag_verifier.data import load_claims
from crag_verifier.fetch import evaluator_path
from crag_verifier.scifact_metrics import (
    bootstrap,
    count_matrix,
    evaluate,
    f1_scores,
    label_counts,
)

FIXTURE = Path(__file__).parent / "fixtures" / "official_parity"


def _load() -> tuple[list, list[dict]]:
    gold = load_claims(FIXTURE / "gold.jsonl")
    preds = [json.loads(line) for line in (FIXTURE / "predictions.jsonl").read_text().splitlines()]
    return gold, preds


def test_matches_recorded_official_output() -> None:
    gold, preds = _load()
    expected = json.loads((FIXTURE / "expected_official_metrics.json").read_text())
    ours = evaluate(gold, preds)
    for key, value in expected.items():
        assert ours[key] == pytest.approx(value, abs=1e-12), key


@pytest.mark.integration
def test_matches_the_official_script_when_fetched(tmp_path: Path) -> None:
    script = evaluator_path()
    if not script.exists():
        pytest.skip("official evaluator not fetched")
    out = tmp_path / "metrics.json"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            str(script),
            "--labels_file",
            str(FIXTURE / "gold.jsonl"),
            "--preds_file",
            str(FIXTURE / "predictions.jsonl"),
            "--metrics_output_file",
            str(out),
            "--verbose",
        ],
        check=True,
    )
    gold, preds = _load()
    ours = evaluate(gold, preds)
    for key, value in json.loads(out.read_text()).items():
        assert ours[key] == pytest.approx(value, abs=1e-12), key


def test_claim_order_must_match() -> None:
    gold, preds = _load()
    with pytest.raises(ValueError):
        count_matrix(gold, list(reversed(preds)))


def test_illegal_labels_are_rejected() -> None:
    gold, preds = _load()
    preds[0] = {"id": preds[0]["id"], "evidence": {"10": {"sentences": [1], "label": "NOINFO"}}}
    with pytest.raises(ValueError):
        evaluate(gold, preds)


def test_bootstrap_intervals_bracket_the_estimate_and_pair_systems() -> None:
    gold, preds = _load()
    counts = count_matrix(gold, preds)
    empty = count_matrix(gold, [{"id": g.id, "evidence": {}} for g in gold])
    result = bootstrap(
        {"base": counts, "same": counts.copy(), "none": empty},
        [("same", "base"), ("none", "base")],
        n_resamples=300,
    )
    point = evaluate(gold, preds)["abstract_label_only_f1"]
    ci = result["intervals"]["base"]["abstract_label_only_f1"]
    assert ci["ci95_low"] <= point <= ci["ci95_high"]
    same = result["differences"]["same - base"]["abstract_label_only_f1"]
    assert same["ci95_low"] == same["ci95_high"] == same["estimate"] == 0.0
    none = result["differences"]["none - base"]["abstract_label_only_f1"]
    assert none["estimate"] == pytest.approx(-point)
    assert result["intervals"]["none"]["abstract_label_only_f1"]["ci95_high"] == 0.0
    assert np.isfinite(result["intervals"]["base"]["sentence_label_f1"]["ci95_low"])


def test_vectorised_bootstrap_matches_scoring_each_resample() -> None:
    gold, preds = _load()
    counts = count_matrix(gold, preds)
    result = bootstrap({"base": counts}, [], n_resamples=50, seed=3)
    rng = np.random.default_rng(3)
    looped = []
    for _ in range(50):
        idx = rng.integers(0, len(gold), size=len(gold))
        looped.append(f1_scores(counts[idx].sum(axis=0))["sentence_label_f1"])
    ci = result["intervals"]["base"]["sentence_label_f1"]
    assert ci["ci95_low"] == pytest.approx(np.percentile(looped, 2.5))
    assert ci["ci95_high"] == pytest.approx(np.percentile(looped, 97.5))


def test_label_counts_split_labelled_abstracts_into_right_and_wrong() -> None:
    gold, preds = _load()
    counts = count_matrix(gold, preds)
    metrics = evaluate(gold, preds)
    every = label_counts(counts)
    labelled = every["correct_labels"] + every["wrong_labels"]
    assert labelled == sum(len(p["evidence"]) for p in preds)
    assert every["correct_labels"] / labelled == pytest.approx(
        metrics["abstract_label_only_precision"]
    )
    halves = [label_counts(counts, range(0, 2)), label_counts(counts, range(2, len(gold)))]
    for key in every:
        assert halves[0][key] + halves[1][key] == every[key]
