from __future__ import annotations

import json

import httpx
import pytest

from crag_verifier.providers import (
    FakeProvider,
    OpenAICompatibleProvider,
    ProviderError,
    provider_from_env,
)
from crag_verifier.rewrite import KeyphraseRewriter, LLMRewriter, candidate_phrases


def test_openai_compatible_request_shape() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "  insulin mice  "}}]})

    provider = OpenAICompatibleProvider(
        "http://llm.local/v1/", "small-model", api_key="k", transport=httpx.MockTransport(handler)
    )
    assert provider.complete("sys", "user", max_tokens=12) == "insulin mice"
    assert seen["url"] == "http://llm.local/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "small-model"
    assert seen["body"]["max_tokens"] == 12 and seen["body"]["temperature"] == 0.0
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]}),
        httpx.Response(200, text="not json"),
    ],
)
def test_provider_failures_raise_provider_error(response: httpx.Response) -> None:
    provider = OpenAICompatibleProvider(
        "http://llm.local/v1", "m", transport=httpx.MockTransport(lambda _: response)
    )
    with pytest.raises(ProviderError):
        provider.complete("s", "u")


def test_fake_provider_is_deterministic() -> None:
    a, b = FakeProvider(), FakeProvider()
    assert a.complete("s", "line one\nline two") == b.complete("s", "line one\nline two")
    assert a.complete("s", "x").endswith("x")


def test_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("CRAG_VERIFIER_LLM", "CRAG_VERIFIER_LLM_BASE_URL", "CRAG_VERIFIER_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert provider_from_env() is None
    monkeypatch.setenv("CRAG_VERIFIER_LLM", "fake")
    assert isinstance(provider_from_env(), FakeProvider)
    monkeypatch.delenv("CRAG_VERIFIER_LLM")
    monkeypatch.setenv("CRAG_VERIFIER_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CRAG_VERIFIER_LLM_MODEL", "local")
    assert isinstance(provider_from_env(), OpenAICompatibleProvider)


def test_candidate_phrases_split_on_stopwords_and_claim_verbs() -> None:
    phrases = candidate_phrases(
        "Vitamin D supplementation reduces the risk of hip fractures in elderly women."
    )
    assert phrases == [
        ["Vitamin", "D", "supplementation"],
        ["risk"],
        ["hip", "fractures"],
        ["elderly", "women"],
    ]


def test_keyphrase_rewrite_keeps_the_rarest_phrases_in_order() -> None:
    idf = {
        "vitamin": 3.0,
        "d": 1.0,
        "supplementation": 2.0,
        "risk": 0.5,
        "hip": 4.0,
        "fracture": 3.5,
        "elderly": 1.0,
        "women": 0.8,
    }
    rewriter = KeyphraseRewriter(lambda t: idf.get(t, 0.0), max_phrases=2)
    assert (
        rewriter.rewrite(
            "Vitamin D supplementation reduces the risk of hip fractures in elderly women."
        )
        == "Vitamin D supplementation hip fractures"
    )


def test_keyphrase_rewrite_of_only_stopwords_returns_the_claim() -> None:
    assert KeyphraseRewriter(lambda t: 1.0).rewrite("It is what it is.") == "It is what it is."


def test_llm_rewriter_uses_provider_and_falls_back() -> None:
    keyphrase = KeyphraseRewriter(lambda t: 1.0)
    assert (
        LLMRewriter(FakeProvider(), keyphrase)
        .rewrite("Insulin lowers glucose.")
        .startswith("[fake ")
    )

    class Broken:
        name = "broken"

        def complete(self, system: str, user: str, *, max_tokens: int = 128) -> str:
            raise ProviderError("down")

    assert LLMRewriter(Broken(), keyphrase).rewrite("Insulin lowers glucose.") == "Insulin glucose"
