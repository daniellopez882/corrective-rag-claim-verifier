"""Command-line guards that must hold without any downloaded data."""

from __future__ import annotations

import pytest

from crag_verifier.cli import HEADLINE_RESULTS, SMOKE_RESULTS, main


def test_smoke_runs_default_to_an_ignored_file_beside_the_headline_results() -> None:
    assert SMOKE_RESULTS.parent == HEADLINE_RESULTS.parent
    assert SMOKE_RESULTS.name.startswith("smoke-")


def test_a_smoke_run_refuses_to_overwrite_the_headline_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["bench", "--smoke", "--out", HEADLINE_RESULTS.as_posix()]) == 2
    assert "smoke-" in capsys.readouterr().err
