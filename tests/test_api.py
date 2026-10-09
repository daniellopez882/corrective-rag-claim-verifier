"""HTTP contract of the service, on the demo backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from crag_verifier.api import MAX_BODY_BYTES, MAX_CLAIM_CHARS, create_app
from crag_verifier.pipeline import Explainer, FittedPolicy, Pipeline, Verifier
from crag_verifier.providers import FakeProvider


@pytest.fixture(scope="module")
def client(demo_pipeline: Pipeline, demo_policy: FittedPolicy) -> TestClient:
    return TestClient(create_app(Verifier(demo_pipeline, demo_policy), "demo", "test policy"))


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "version": body["version"],
        "backend": "demo",
        "corpus_documents": 12,
        "policy": "test policy",
    }


def test_verify_returns_the_documented_shape(client: TestClient) -> None:
    resp = client.post("/verify", json={"claim": "Strain K4 degrades polyethylene film."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] in {"SUPPORT", "CONTRADICT", "NOINFO"}
    assert body["action"] in {"correct", "ambiguous", "incorrect"}
    assert 0.0 <= body["evaluator_confidence"] <= 1.0
    assert body["abstained"] == (body["action"] == "incorrect")
    assert body["explanation_status"] == "not_requested"
    assert [t["step"] for t in body["trace"]][:3] == ["retrieve", "nli", "evaluate"]
    for ev in body["evidence"]:
        assert [s["id"] for s in ev["rationale"]] == ev["rationale_sentence_ids"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"claim": "ab"},
        {"claim": "x" * (MAX_CLAIM_CHARS + 1)},
        {"claim": "A valid claim.", "system": "magic"},
        {"claim": "A valid claim.", "unexpected": 1},
        {"claim": 12345},
        {"claim": "     "},
    ],
)
def test_invalid_requests_get_422(client: TestClient, payload: dict) -> None:
    assert client.post("/verify", json=payload).status_code == 422


def test_declared_oversized_body_gets_413(client: TestClient) -> None:
    payload = '{"claim": "' + "a" * MAX_BODY_BYTES + '"}'
    resp = client.post("/verify", content=payload, headers={"content-type": "application/json"})
    assert resp.status_code == 413


def test_streamed_oversized_body_gets_413(client: TestClient) -> None:
    def chunks():
        yield b'{"claim": "'
        for _ in range(10):
            yield b"a" * 1024
        yield b'"}'

    resp = client.post("/verify", content=chunks(), headers={"content-type": "application/json"})
    assert resp.status_code == 413


def test_openapi_documents_verify(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/verify" in schema["paths"]
    props = schema["components"]["schemas"]["VerifyResponse"]["properties"]
    for field in ("verdict", "action", "evaluator_confidence", "trace", "evidence"):
        assert field in props


def test_explain_without_provider_says_so(client: TestClient) -> None:
    body = client.post(
        "/verify", json={"claim": "Velorin lowers fasting blood glucose in mice.", "explain": True}
    ).json()
    assert body["explanation"] is None and body["explanation_status"] == "no_provider"


def test_explain_with_the_fake_provider(demo_pipeline: Pipeline, demo_policy: FittedPolicy) -> None:
    verifier = Verifier(demo_pipeline, demo_policy, Explainer(FakeProvider()))
    with TestClient(create_app(verifier, "demo", "test")) as c:
        body = c.post(
            "/verify",
            json={"claim": "Velorin lowers fasting blood glucose in mice.", "explain": True},
        ).json()
    assert body["explanation_status"] == "ok"
    assert body["explanation"].startswith("[fake ")
