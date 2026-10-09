"""Real-model checks; run with `pytest -m integration` after `crag-verifier fetch`."""

from __future__ import annotations

import numpy as np
import pytest

from crag_verifier.models import CONTRADICT, ENTAIL

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def models():
    from crag_verifier.runtime import SetupError, load_models

    try:
        return load_models("onnx")
    except SetupError as exc:
        pytest.skip(str(exc))


def test_nli_separates_entailment_and_contradiction(models) -> None:
    probs = models.nli.predict(
        [
            ("A man is playing a guitar on stage.", "A man is playing an instrument."),
            ("A man is playing a guitar on stage.", "Nobody is playing music."),
        ]
    )
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert probs[0, ENTAIL] > 0.8 and probs[1, CONTRADICT] > 0.8


def test_cross_encoder_prefers_the_relevant_passage(models) -> None:
    scores = models.cross_encoder.score(
        [
            (
                "Does metformin lower blood glucose?",
                "Metformin reduced fasting glucose in patients.",
            ),
            ("Does metformin lower blood glucose?", "The bridge was painted red in 1937."),
        ]
    )
    assert scores[0] > scores[1]


def test_dense_vectors_are_unit_length(models) -> None:
    q = models.encoder.encode_queries(["insulin resistance"])
    d = models.encoder.encode_documents(
        ["Insulin resistance in muscle.", "Volcanic soil chemistry."]
    )
    assert q.shape == (1, 384) and d.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, rtol=1e-5)
    assert (d @ q[0])[0] > (d @ q[0])[1]
