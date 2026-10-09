from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from crag_verifier.calibration import expected_calibration_error, reliability_bins
from crag_verifier.reliability_svg import reliability_svg

NS = "{http://www.w3.org/2000/svg}"


def test_svg_is_well_formed_with_one_curve_per_method() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    methods = {
        name: {
            "reliability": reliability_bins(q, y),
            "ece_10bins": expected_calibration_error(q, y),
        }
        for name, q in {"raw": p, "platt": np.sqrt(p)}.items()
    }
    root = ET.fromstring(reliability_svg(methods, title="t <& test>"))
    assert root.tag == f"{NS}svg"
    assert len(root.findall(f"{NS}polyline")) == 2
    texts = [t.text for t in root.iter(f"{NS}text")]
    assert "t <& test>" in texts
    assert any(t and t.startswith("platt (ECE ") for t in texts)


def test_empty_bins_are_skipped() -> None:
    y = np.array([0, 1])
    methods = {
        "raw": {"reliability": reliability_bins(np.array([0.05, 0.95]), y), "ece_10bins": 0.05}
    }
    root = ET.fromstring(reliability_svg(methods, title="x"))
    assert len(root.findall(f"{NS}circle")) == 2
