"""LLM provider interface used only for the optional rewrite and explanation steps.

The adapter speaks the OpenAI-compatible ``/chat/completions`` protocol, which OpenAI,
Ollama (``http://localhost:11434/v1``), vLLM and LM Studio all serve. Nothing in the
benchmark depends on it; the fake provider keeps tests and offline runs deterministic.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import httpx


class ProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, *, max_tokens: int = 128) -> str: ...


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.name = f"openai-compatible:{model}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def complete(self, system: str, user: str, *, max_tokens: int = 128) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError(f"{self.name}: empty completion")
        return content.strip()


class FakeProvider:
    """Deterministic stand-in: echoes the last line of the prompt with a stable tag."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, max_tokens: int = 128) -> str:
        self.calls.append((system, user))
        tag = hashlib.sha256((system + "\x1f" + user).encode()).hexdigest()[:8]
        last = user.strip().splitlines()[-1] if user.strip() else ""
        return f"[fake {tag}] {last}"[: max_tokens * 4]


def provider_from_env() -> LLMProvider | None:
    """CRAG_VERIFIER_LLM=fake, or CRAG_VERIFIER_LLM_BASE_URL + CRAG_VERIFIER_LLM_MODEL."""
    if os.environ.get("CRAG_VERIFIER_LLM", "").lower() == "fake":
        return FakeProvider()
    base_url = os.environ.get("CRAG_VERIFIER_LLM_BASE_URL")
    model = os.environ.get("CRAG_VERIFIER_LLM_MODEL")
    if base_url and model:
        return OpenAICompatibleProvider(
            base_url, model, api_key=os.environ.get("CRAG_VERIFIER_LLM_API_KEY")
        )
    return None
