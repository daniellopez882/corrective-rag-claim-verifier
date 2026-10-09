"""The corrective RAG decision flow and the four systems it is compared as."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from crag_verifier.aggregate import (
    AggregatorParams,
    DocDecision,
    DocEvidence,
    EvidenceMode,
    Verdict,
    label_doc,
    sentence_evidence,
    strip_evidence,
    verdict,
)
from crag_verifier.data import Corpus
from crag_verifier.evaluator import FEATURES, RetrievalEvaluator, evaluator_features
from crag_verifier.models import NLIModel, PairScorer
from crag_verifier.policy import Action, Thresholds, decide, thresholds_from_dict
from crag_verifier.providers import LLMProvider, ProviderError
from crag_verifier.retrieval import HybridRetriever, RetrievalResult
from crag_verifier.rewrite import QueryRewriter
from crag_verifier.strips import Strip, decompose, rationale_ids, recompose, select


class System(StrEnum):
    STANDARD_RAG = "standard_rag"
    ALWAYS_EXPAND = "always_expand"
    CRAG_NO_STRIPS = "crag_no_strips"
    CRAG = "crag"


@dataclass(frozen=True)
class SystemSpec:
    triage: bool  # let the calibrated evaluator choose the action
    fixed_action: Action  # the action taken when triage is off
    evidence: EvidenceMode
    description: str


SYSTEMS: dict[System, SystemSpec] = {
    System.STANDARD_RAG: SystemSpec(
        False, Action.CORRECT, EvidenceMode.SENTENCES, "top-3 abstracts, sentence-level NLI"
    ),
    System.ALWAYS_EXPAND: SystemSpec(
        False,
        Action.AMBIGUOUS,
        EvidenceMode.SENTENCES,
        "always rewrite + dense re-retrieve, sentence-level NLI",
    ),
    System.CRAG_NO_STRIPS: SystemSpec(
        True, Action.CORRECT, EvidenceMode.SENTENCES, "evaluator triage, sentence-level NLI"
    ),
    System.CRAG: SystemSpec(
        True, Action.CORRECT, EvidenceMode.STRIPS, "evaluator triage, strip refinement"
    ),
}


@dataclass
class FittedPolicy:
    evaluator: RetrievalEvaluator
    thresholds: dict[EvidenceMode, Thresholds]
    aggregator: dict[EvidenceMode, AggregatorParams]
    strip_min_relevance: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": 1,
            "evaluator": self.evaluator.to_dict(),
            "thresholds": {m.value: t.to_dict() for m, t in self.thresholds.items()},
            "aggregator": {m.value: a.to_dict() for m, a in self.aggregator.items()},
            "strip_min_relevance": self.strip_min_relevance,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FittedPolicy:
        return cls(
            evaluator=RetrievalEvaluator.from_dict(d["evaluator"]),
            thresholds={
                EvidenceMode(k): thresholds_from_dict(v) for k, v in d["thresholds"].items()
            },
            aggregator={
                EvidenceMode(k): AggregatorParams.from_dict(v) for k, v in d["aggregator"].items()
            },
            strip_min_relevance=float(d["strip_min_relevance"]),
            provenance=d.get("provenance", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FittedPolicy:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class TraceStep:
    step: str
    ms: float
    detail: dict[str, Any]


@dataclass
class InitialAnalysis:
    retrieval: RetrievalResult
    sentence_probs: dict[int, np.ndarray]
    features: np.ndarray


@dataclass
class Verification:
    claim: str
    system: System
    action: Action
    confidence: float
    verdict: Verdict
    verdict_probability: float
    decisions: list[DocDecision]
    rewritten_query: str | None
    trace: list[TraceStep]
    explanation: str | None = None

    def prediction(self, claim_id: int) -> dict[str, Any]:
        """The record the SciFact evaluator reads: labelled abstracts with rationales."""
        return {
            "id": claim_id,
            "evidence": {
                str(d.doc_id): {"sentences": list(d.rationale), "label": d.label}
                for d in self.decisions
                if d.label is not None
            },
        }


class Pipeline:
    """Policy-free building blocks shared by the service, the fit and the benchmark."""

    def __init__(
        self,
        corpus: Corpus,
        retriever: HybridRetriever,
        cross_encoder: PairScorer,
        nli: NLIModel,
        rewriter: QueryRewriter,
        *,
        top_n: int = 3,
        max_sentences: int = 64,
        strip_min_words: int = 4,
    ):
        self.corpus = corpus
        self.retriever = retriever
        self.cross_encoder = cross_encoder
        self.nli = nli
        self.rewriter = rewriter
        self.top_n = top_n
        self.max_sentences = max_sentences
        self.strip_min_words = strip_min_words

    def sentence_nli(self, claim: str, doc_ids: Sequence[int]) -> dict[int, np.ndarray]:
        """(n_sentences, 3) NLI per abstract: each sentence as premise, the claim as hypothesis."""
        spans: list[tuple[int, int, int]] = []
        pairs: list[tuple[str, str]] = []
        for doc_id in doc_ids:
            sentences = self.corpus[doc_id].sentences[: self.max_sentences]
            spans.append((doc_id, len(pairs), len(pairs) + len(sentences)))
            pairs.extend((s, claim) for s in sentences)
        probs = self.nli.predict(pairs)
        return {doc_id: probs[lo:hi] for doc_id, lo, hi in spans}

    def featurise(self, claim: str, retrieval: RetrievalResult) -> InitialAnalysis:
        probs = self.sentence_nli(claim, retrieval.top(self.top_n))
        return InitialAnalysis(retrieval, probs, evaluator_features(retrieval, probs, self.top_n))

    def analyse(self, claim: str) -> InitialAnalysis:
        return self.featurise(claim, self.retriever.retrieve(claim))

    def expand(self, claim: str, retrieval: RetrievalResult) -> tuple[str, RetrievalResult]:
        rewritten = self.rewriter.rewrite(claim)
        return rewritten, self.retriever.expand(claim, rewritten, retrieval)

    def strip_relevance(
        self, claim: str, doc_ids: Sequence[int]
    ) -> dict[int, tuple[list[Strip], np.ndarray]]:
        strips = {
            d: decompose(self.corpus[d], self.strip_min_words, self.max_sentences) for d in doc_ids
        }
        pairs = [(claim, s.text) for d in doc_ids for s in strips[d]]
        scores = self.cross_encoder.score(pairs)
        out: dict[int, tuple[list[Strip], np.ndarray]] = {}
        offset = 0
        for d in doc_ids:
            n = len(strips[d])
            out[d] = (strips[d], scores[offset : offset + n])
            offset += n
        return out

    def evidence(
        self,
        claim: str,
        doc_ids: Sequence[int],
        mode: EvidenceMode,
        min_relevance: float,
        sentence_probs: dict[int, np.ndarray] | None = None,
    ) -> list[DocEvidence]:
        if mode is EvidenceMode.SENTENCES:
            known = dict(sentence_probs or {})
            missing = [d for d in doc_ids if d not in known]
            known.update(self.sentence_nli(claim, missing))
            return [sentence_evidence(d, known[d]) for d in doc_ids]
        relevance = self.strip_relevance(claim, doc_ids)
        kept = {d: select(*relevance[d], min_relevance) for d in doc_ids}
        readable = [d for d in doc_ids if kept[d]]
        passage_probs = self.nli.predict([(recompose(kept[d]), claim) for d in readable])
        by_doc = dict(zip(readable, passage_probs, strict=True))
        return [strip_evidence(d, by_doc.get(d), rationale_ids(kept[d])) for d in doc_ids]


class Explainer:
    SYSTEM = (
        "You explain a claim-verification verdict in two sentences for a scientist. "
        "Use only the evidence sentences given; do not add facts."
    )

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def explain(self, corpus: Corpus, result: Verification) -> str:
        lines = [f"Claim: {result.claim}", f"Verdict: {result.verdict}"]
        for d in result.decisions:
            if d.label is None:
                continue
            doc = corpus[d.doc_id]
            for sid in d.rationale:
                lines.append(f"[{d.doc_id}:{sid}] ({d.label}) {doc.sentences[sid]}")
        lines.append("Explain the verdict.")
        return self.provider.complete(self.SYSTEM, "\n".join(lines), max_tokens=160)


class Verifier:
    def __init__(
        self, pipeline: Pipeline, policy: FittedPolicy, explainer: Explainer | None = None
    ):
        self.pipeline = pipeline
        self.policy = policy
        self.explainer = explainer

    def verify(
        self, claim: str, system: System = System.CRAG, explain: bool = False
    ) -> Verification:
        spec = SYSTEMS[system]
        pipe = self.pipeline
        trace: list[TraceStep] = []
        clock = time.perf_counter()

        def mark(step: str, **detail: Any) -> None:
            nonlocal clock
            now = time.perf_counter()
            trace.append(TraceStep(step, round((now - clock) * 1000.0, 2), detail))
            clock = now

        retrieval = pipe.retriever.retrieve(claim)
        mark(
            "retrieve",
            top=[
                {"doc_id": c.doc_id, "ce_score": round(c.ce_score, 4)}
                for c in retrieval.candidates[: pipe.top_n]
            ],
            pool_size=len(retrieval.candidates),
        )
        initial = pipe.featurise(claim, retrieval)
        mark("nli", sentences=sum(len(p) for p in initial.sentence_probs.values()))
        confidence = float(self.policy.evaluator.confidence(initial.features[None, :])[0])
        thresholds = self.policy.thresholds[spec.evidence]
        action = decide(confidence, thresholds) if spec.triage else spec.fixed_action
        mark(
            "evaluate",
            features={
                n: round(float(v), 4) for n, v in zip(FEATURES, initial.features, strict=True)
            },
            confidence=round(confidence, 4),
            thresholds=thresholds.to_dict(),
            action=action.value,
            triage=spec.triage,
        )

        rewritten: str | None = None
        decisions: list[DocDecision] = []
        if action is Action.INCORRECT:
            mark("abstain", reason="evaluator judged the retrieval wrong; answering NOINFO")
        else:
            if action is Action.AMBIGUOUS:
                rewritten, retrieval = pipe.expand(claim, retrieval)
                mark(
                    "re-retrieve",
                    rewritten_query=rewritten,
                    top=[c.doc_id for c in retrieval.candidates[: pipe.top_n]],
                    pool_size=len(retrieval.candidates),
                )
            doc_ids = retrieval.top(pipe.top_n)
            evidence = pipe.evidence(
                claim,
                doc_ids,
                spec.evidence,
                self.policy.strip_min_relevance,
                initial.sentence_probs,
            )
            strips = spec.evidence is EvidenceMode.STRIPS
            mark(
                "refine" if strips else "read",
                mode=spec.evidence.value,
                abstracts={
                    str(e.doc_id): {
                        "entail": round(e.entail, 4),
                        "contradict": round(e.contradict, 4),
                        **({"kept_sentence_ids": list(e.support_rationale)} if strips else {}),
                    }
                    for e in evidence
                },
            )
            params = self.policy.aggregator[spec.evidence]
            decisions = [label_doc(e, params) for e in evidence]
            mark(
                "aggregate",
                labels={str(d.doc_id): d.label or "NOINFO" for d in decisions},
                params=params.to_dict(),
            )
        label, probability = verdict(decisions)
        result = Verification(
            claim, system, action, confidence, label, probability, decisions, rewritten, trace
        )
        if explain and self.explainer is not None:
            try:
                result.explanation = self.explainer.explain(pipe.corpus, result)
                mark("explain", provider=self.explainer.provider.name)
            except ProviderError as exc:
                mark("explain", error=str(exc))
        return result
