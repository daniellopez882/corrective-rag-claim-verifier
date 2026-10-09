"""Pinned external artifacts: where they come from and the sha256 they must hash to.

The checksums were recorded on the first download (2026-10-09). Model files come from
the Hugging Face Hub at a fixed commit; ONNX weights carry the LFS sha256 published by
the Hub, which matched the bytes we received.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_ENV = "CRAG_VERIFIER_DATA_DIR"


def data_dir() -> Path:
    """Root for SciFact, the official evaluator, indexes and the score cache."""
    raw = os.environ.get(DATA_DIR_ENV)
    root = Path(raw) if raw else Path.home() / ".cache" / "corrective-rag-claim-verifier"
    return root.expanduser()


@dataclass(frozen=True)
class RemoteFile:
    url: str
    sha256: str
    filename: str


SCIFACT_ARCHIVE = RemoteFile(
    url="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
    sha256="11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be",
    filename="data.tar.gz",
)

# Files taken out of the archive; everything else in it is ignored.
SCIFACT_FILES: dict[str, str] = {
    "corpus.jsonl": "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62",
    "claims_train.jsonl": "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0",
    "claims_dev.jsonl": "86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217",
    "claims_test.jsonl": "558930d75215c73f84a28fe538307d6d397c9de1ec7239514cf45f80d75d2ca3",
}

# Leaderboard evaluator from allenai/scifact-evaluator (Apache-2.0), pinned to a commit.
OFFICIAL_EVALUATOR = RemoteFile(
    url=(
        "https://raw.githubusercontent.com/allenai/scifact-evaluator/"
        "66feffc5b2cc9e28e3ce3b8c9e824c3c642981eb/evaluator/eval.py"
    ),
    sha256="2554fed44c3f5592bdeed59ab0d3918a412b452ba1c77da73ba0f62e74c9987f",
    filename="eval.py",
)


@dataclass(frozen=True)
class HubModel:
    """An ONNX export on the Hugging Face Hub plus the files needed to run it."""

    key: str
    repo_id: str
    revision: str
    onnx_file: str
    licence: str
    sha256: dict[str, str]

    @property
    def files(self) -> list[str]:
        return list(self.sha256)


DENSE_MODEL = HubModel(
    key="dense",
    repo_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    onnx_file="onnx/model.onnx",
    licence="MIT",
    sha256={
        "onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "config.json": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    },
)

CROSS_ENCODER_MODEL = HubModel(
    key="cross_encoder",
    repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    onnx_file="onnx/model_quint8_avx2.onnx",
    licence="Apache-2.0",
    sha256={
        "onnx/model_quint8_avx2.onnx": (
            "c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9"
        ),
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "config.json": "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    },
)

# The spec named xtremedistil-l6-h256; every public checkpoint of it is binary
# (entailment / not-entailment), and the evaluator needs a contradiction probability.
NLI_MODEL = HubModel(
    key="nli",
    repo_id="cross-encoder/nli-MiniLM2-L6-H768",
    revision="b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
    onnx_file="onnx/model_quint8_avx2.onnx",
    licence="Apache-2.0",
    sha256={
        "onnx/model_quint8_avx2.onnx": (
            "44391a5241a62e0083c1a8899a71e69a092b95aea5ba89e14062925468eceac7"
        ),
        "tokenizer.json": "82139106e603ee4e1d5bc99d056ccbed5a92bc24848b1b5a7137c26e00d0dbf6",
        "config.json": "8b0e41caff7567c0f53e6983f35591c3dec59507c9173ab125c5823394fb57f3",
    },
)

ALL_MODELS: tuple[HubModel, ...] = (DENSE_MODEL, CROSS_ENCODER_MODEL, NLI_MODEL)
