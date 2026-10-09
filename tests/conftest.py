from __future__ import annotations

import pytest

from crag_verifier.data import Claim, Corpus
from crag_verifier.fit import fit_policy
from crag_verifier.pipeline import FittedPolicy, Pipeline
from crag_verifier.runtime import build_pipeline, load_demo, load_models


@pytest.fixture(scope="session")
def demo_data() -> tuple[Corpus, list[Claim]]:
    return load_demo()


@pytest.fixture(scope="session")
def demo_pipeline(demo_data: tuple[Corpus, list[Claim]]) -> Pipeline:
    corpus, _ = demo_data
    pipeline, _ = build_pipeline(corpus, load_models("demo"), None)
    return pipeline


@pytest.fixture(scope="session")
def demo_policy(demo_pipeline: Pipeline, demo_data: tuple[Corpus, list[Claim]]) -> FittedPolicy:
    _, claims = demo_data
    policy, _, _ = fit_policy(demo_pipeline, claims, folds=3)
    return policy
