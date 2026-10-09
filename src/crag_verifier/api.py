"""FastAPI service: POST /verify, GET /health."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from crag_verifier import __version__
from crag_verifier.pipeline import Explainer, System, Verification, Verifier
from crag_verifier.policy import Action
from crag_verifier.providers import provider_from_env

MAX_BODY_BYTES = 8 * 1024
MAX_CLAIM_CHARS = 1000


class BodyLimit:
    """Reject request bodies over ``limit`` bytes with 413, declared or streamed."""

    def __init__(self, app: ASGIApp, limit: int = MAX_BODY_BYTES):
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length" and value.isdigit() and int(value) > self.limit:
                await self._reject(send)
                return
        received = 0
        body_chunks: list[bytes] = []
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunk = message.get("body", b"")
            received += len(chunk)
            if received > self.limit:
                await self._reject(send)
                return
            body_chunks.append(chunk)
            more = message.get("more_body", False)
        body = b"".join(body_chunks)
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, send: Send) -> None:
        payload = b'{"detail":"request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=3, max_length=MAX_CLAIM_CHARS, description="Claim to verify")
    system: System = Field(System.CRAG, description="Which pipeline variant to run")
    explain: bool = Field(False, description="Ask the configured LLM provider for an explanation")


class SentenceOut(BaseModel):
    id: int
    text: str


class EvidenceOut(BaseModel):
    doc_id: int
    title: str
    label: Literal["SUPPORT", "CONTRADICT", "NOINFO"]
    probability: float
    rationale_sentence_ids: list[int]
    rationale: list[SentenceOut]


class TraceOut(BaseModel):
    step: str
    ms: float
    detail: dict[str, Any]


class VerifyResponse(BaseModel):
    claim: str
    system: System
    verdict: Literal["SUPPORT", "CONTRADICT", "NOINFO"]
    verdict_probability: float
    action: Literal["correct", "ambiguous", "incorrect"]
    abstained: bool
    evaluator_confidence: float = Field(
        description="Calibrated probability that a gold abstract is in the top-3 retrieved"
    )
    rewritten_query: str | None
    evidence: list[EvidenceOut]
    explanation: str | None
    explanation_status: Literal["not_requested", "no_provider", "ok", "failed"]
    trace: list[TraceOut]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    backend: str
    corpus_documents: int
    policy: str


def to_response(result: Verification, verifier: Verifier, explain: bool) -> VerifyResponse:
    corpus = verifier.pipeline.corpus
    evidence = []
    for d in result.decisions:
        doc = corpus[d.doc_id]
        evidence.append(
            EvidenceOut(
                doc_id=d.doc_id,
                title=doc.title,
                label=d.label or "NOINFO",
                probability=round(d.probability, 4),
                rationale_sentence_ids=list(d.rationale),
                rationale=[SentenceOut(id=i, text=doc.sentences[i]) for i in d.rationale],
            )
        )
    if not explain:
        status = "not_requested"
    elif verifier.explainer is None:
        status = "no_provider"
    else:
        status = "ok" if result.explanation is not None else "failed"
    return VerifyResponse(
        claim=result.claim,
        system=result.system,
        verdict=result.verdict,
        verdict_probability=round(result.verdict_probability, 4),
        action=result.action.value,
        abstained=result.action is Action.INCORRECT,
        evaluator_confidence=round(result.confidence, 4),
        rewritten_query=result.rewritten_query,
        evidence=evidence,
        explanation=result.explanation,
        explanation_status=status,
        trace=[TraceOut(step=t.step, ms=t.ms, detail=t.detail) for t in result.trace],
    )


def create_app(verifier: Verifier, backend: str, policy_source: str) -> FastAPI:
    app = FastAPI(
        title="Corrective RAG claim verifier",
        version=__version__,
        description=(
            "Verifies a scientific claim against the SciFact corpus. A calibrated retrieval "
            "evaluator chooses to refine the retrieved abstracts, re-retrieve with a "
            "rewritten query, or abstain with NOINFO."
        ),
    )
    # One verification at a time: the ONNX sessions already use every core.
    lock = asyncio.Lock()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            backend=backend,
            corpus_documents=len(verifier.pipeline.corpus),
            policy=policy_source,
        )

    @app.post("/verify", response_model=VerifyResponse)
    async def verify(request: VerifyRequest) -> VerifyResponse:
        if not request.claim.strip():
            raise HTTPException(status_code=422, detail="claim is blank")
        async with lock:
            result = await asyncio.to_thread(
                verifier.verify, request.claim.strip(), request.system, request.explain
            )
        return to_response(result, verifier, request.explain)

    app.add_middleware(BodyLimit, limit=MAX_BODY_BYTES)
    return app


def app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory``; backend from CRAG_VERIFIER_BACKEND (default onnx)."""
    from crag_verifier.runtime import service_pipeline

    backend = os.environ.get("CRAG_VERIFIER_BACKEND", "onnx")
    pipeline, policy, source = service_pipeline(backend)
    provider = provider_from_env()
    verifier = Verifier(pipeline, policy, Explainer(provider) if provider else None)
    return create_app(verifier, backend, source)
