"""Wiring: build the pipeline for a backend and load (or fit) its policy."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from crag_verifier.artifacts import data_dir
from crag_verifier.data import Claim, Corpus, load_claims, load_corpus
from crag_verifier.fakes import HashingEncoder, LexicalNLI, OverlapCrossEncoder
from crag_verifier.fetch import scifact_dir
from crag_verifier.models import DenseEncoder, NLIModel, PairScorer
from crag_verifier.pipeline import FittedPolicy, Pipeline
from crag_verifier.providers import provider_from_env
from crag_verifier.retrieval import HybridRetriever
from crag_verifier.rewrite import KeyphraseRewriter, LLMRewriter, QueryRewriter

BACKENDS = ("onnx", "demo")


class SetupError(RuntimeError):
    """Something the service needs is missing; the message says which command creates it."""


@dataclass
class ModelSet:
    encoder: DenseEncoder
    cross_encoder: PairScorer
    nli: NLIModel


def demo_paths() -> tuple[Path, Path]:
    root = resources.files("crag_verifier") / "demo"
    return Path(str(root / "corpus.jsonl")), Path(str(root / "claims.jsonl"))


def load_demo() -> tuple[Corpus, list[Claim]]:
    corpus_path, claims_path = demo_paths()
    return load_corpus(corpus_path), load_claims(claims_path)


def corpus_fingerprint(corpus: Corpus) -> str:
    digest = hashlib.sha256()
    for doc in corpus:
        digest.update(f"{doc.doc_id}\x1e{doc.text}\x1d".encode())
    return digest.hexdigest()


def dense_index_path(corpus: Corpus, encoder: DenseEncoder, cache_dir: Path) -> Path:
    tag = hashlib.sha256(f"{encoder.name}|{corpus_fingerprint(corpus)}".encode()).hexdigest()
    return cache_dir / f"dense-{tag[:20]}.npy"


def dense_vectors(
    corpus: Corpus, encoder: DenseEncoder, cache_dir: Path | None, batch: int = 64
) -> tuple[np.ndarray, float]:
    """Document embeddings, cached on disk per (model, corpus); returns (matrix, seconds)."""
    path = None
    if cache_dir is not None:
        path = dense_index_path(corpus, encoder, cache_dir)
        if path.exists():
            matrix = np.load(path)
            if matrix.shape[0] == len(corpus):
                return matrix, 0.0
    start = time.perf_counter()
    texts = [d.text for d in corpus]
    chunks = []
    for i in range(0, len(texts), batch):
        chunks.append(encoder.encode_documents(texts[i : i + batch]))
        done = min(i + batch, len(texts))
        if len(texts) > 4 * batch and (done // batch) % 10 == 0:
            print(f"[index] {done}/{len(texts)} abstracts encoded", file=sys.stderr, flush=True)
    matrix = np.concatenate(chunks).astype(np.float32)
    seconds = time.perf_counter() - start
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
    return matrix, seconds


def load_models(backend: str) -> ModelSet:
    if backend == "demo":
        return ModelSet(HashingEncoder(), OverlapCrossEncoder(), LexicalNLI())
    if backend == "onnx":
        from crag_verifier.models import load_onnx_models

        try:
            encoder, cross_encoder, nli = load_onnx_models()
        except Exception as exc:  # huggingface_hub raises several types for a cold cache
            raise SetupError(
                f"ONNX models not in the local cache; run `crag-verifier fetch` ({exc})"
            ) from exc
        return ModelSet(encoder, cross_encoder, nli)
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


def load_scifact_corpus(root: Path | None = None) -> Corpus:
    path = scifact_dir(root) / "corpus.jsonl"
    if not path.exists():
        raise SetupError(f"{path} is missing; run `crag-verifier fetch`")
    return load_corpus(path)


def make_rewriter(retriever: HybridRetriever) -> QueryRewriter:
    keyphrase = KeyphraseRewriter(retriever.bm25.idf)
    if os.environ.get("CRAG_VERIFIER_REWRITER", "keyphrase") == "llm":
        provider = provider_from_env()
        if provider is None:
            raise SetupError("CRAG_VERIFIER_REWRITER=llm needs an LLM provider configured")
        return LLMRewriter(provider, keyphrase)
    return keyphrase


def build_pipeline(
    corpus: Corpus, models: ModelSet, index_dir: Path | None
) -> tuple[Pipeline, float]:
    vectors, seconds = dense_vectors(corpus, models.encoder, index_dir)
    retriever = HybridRetriever(corpus, models.encoder, models.cross_encoder, vectors)
    pipeline = Pipeline(
        corpus, retriever, models.cross_encoder, models.nli, make_rewriter(retriever)
    )
    return pipeline, seconds


def policy_path() -> Path:
    raw = os.environ.get("CRAG_VERIFIER_POLICY")
    return Path(raw) if raw else data_dir() / "policy.json"


def service_pipeline(backend: str) -> tuple[Pipeline, FittedPolicy, str]:
    """Pipeline and policy for the API; the demo backend fits its policy on the demo claims."""
    models = load_models(backend)
    if backend == "demo":
        from crag_verifier.fit import fit_policy

        corpus, claims = load_demo()
        pipeline, _ = build_pipeline(corpus, models, None)
        policy, _, _ = fit_policy(pipeline, claims, folds=3)
        return pipeline, policy, "fitted at startup on the synthetic demo claims"
    corpus = load_scifact_corpus()
    index_dir = data_dir() / "index"
    if not dense_index_path(corpus, models.encoder, index_dir).exists():
        raise SetupError("dense index missing; run `crag-verifier index`")
    path = policy_path()
    if not path.exists():
        raise SetupError(f"{path} is missing; run `crag-verifier fit` (or the benchmark)")
    pipeline, _ = build_pipeline(corpus, models, index_dir)
    return pipeline, FittedPolicy.load(path), str(path)
