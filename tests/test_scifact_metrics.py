from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from crag_verifier.data import load_claims
from crag_verifier.fetch import evaluator_path
from crag_verifier.scifact_metrics import bootstrap, count_matrix, evaluate

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
        {"base": counts, "same": counts.copy(), "none": empty}, "base", n_resamples=300
    )
    point = evaluate(gold, preds)["abstract_label_only_f1"]
    ci = result["base"]["abstract_label_only_f1"]
    assert ci["ci95_low"] <= point <= ci["ci95_high"]
    same = result["same"]["abstract_label_only_f1"]
    assert same["diff_vs_baseline_ci95_low"] == same["diff_vs_baseline_ci95_high"] == 0.0
    assert result["none"]["abstract_label_only_f1"]["ci95_high"] == 0.0
    assert np.isfinite(result["base"]["sentence_label_f1"]["ci95_low"])
