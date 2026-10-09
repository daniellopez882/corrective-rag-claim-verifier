"""Model interfaces and their ONNX Runtime implementations.

Every model the pipeline touches sits behind one of three small protocols so the
tests, the CI smoke run and the demo backend can swap in the lexical doubles from
``crag_verifier.fakes`` without touching pipeline code.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from crag_verifier.artifacts import CROSS_ENCODER_MODEL, DENSE_MODEL, NLI_MODEL, HubModel

# Column order of every NLI probability matrix in this package.
ENTAIL, NEUTRAL, CONTRADICT = 0, 1, 2

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DenseEncoder(Protocol):
    name: str

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...


class PairScorer(Protocol):
    """Relevance cross-encoder: one logit per (query, passage) pair."""

    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


class NLIModel(Protocol):
    """Premise/hypothesis classifier returning [entail, neutral, contradict] probabilities."""

    name: str

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class _OnnxTextModel:
    """Tokenizer + ONNX Runtime session with length-sorted batching."""

    def __init__(self, model: HubModel, files: dict[str, Path], max_length: int, batch: int):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model = model
        self.name = f"{model.repo_id}@{model.revision[:12]}:{Path(model.onnx_file).name}"
        self.batch = batch
        self.tokenizer = Tokenizer.from_file(str(files["tokenizer.json"]))
        self.tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
        self.tokenizer.no_padding()
        opts = ort.SessionOptions()
        threads = int(os.environ.get("CRAG_VERIFIER_THREADS", "0"))
        if threads > 0:
            opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(files[model.onnx_file]), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.config = json.loads(Path(files["config.json"]).read_text(encoding="utf-8"))
        pad = self.config.get("pad_token_id")
        self.pad_id = int(pad) if pad is not None else 0

    def _run(self, encodings: list) -> list[np.ndarray]:
        order = np.argsort([len(e.ids) for e in encodings], kind="stable")
        outputs: list[np.ndarray | None] = [None] * len(encodings)
        for start in range(0, len(order), self.batch):
            idx = order[start : start + self.batch]
            width = max(len(encodings[i].ids) for i in idx)
            ids = np.full((len(idx), width), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            types = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                enc = encodings[i]
                n = len(enc.ids)
                ids[row, :n] = enc.ids
                mask[row, :n] = 1
                types[row, :n] = enc.type_ids
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = types
            result = self.session.run(None, feed)[0]
            for row, i in enumerate(idx):
                outputs[i] = result[row]
        return [o for o in outputs if o is not None]


class OnnxDenseEncoder(_OnnxTextModel):
    """bge-small-en-v1.5: CLS pooling, L2-normalised, instruction prefix on queries."""

    def __init__(self, files: dict[str, Path], max_length: int = 512, batch: int = 16):
        super().__init__(DENSE_MODEL, files, max_length, batch)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, int(self.config["hidden_size"])), dtype=np.float32)
        encodings = self.tokenizer.encode_batch(list(texts))
        hidden = self._run(encodings)
        cls = np.stack([h[0] for h in hidden]).astype(np.float32)
        return cls / np.linalg.norm(cls, axis=1, keepdims=True).clip(min=1e-12)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([BGE_QUERY_INSTRUCTION + t for t in texts])

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


class OnnxCrossEncoder(_OnnxTextModel):
    def __init__(self, files: dict[str, Path], max_length: int = 256, batch: int = 32):
        super().__init__(CROSS_ENCODER_MODEL, files, max_length, batch)

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        encodings = self.tokenizer.encode_batch([list(p) for p in pairs])
        return np.array([float(o.reshape(-1)[0]) for o in self._run(encodings)], dtype=np.float32)


class OnnxNLI(_OnnxTextModel):
    def __init__(self, files: dict[str, Path], max_length: int = 256, batch: int = 32):
        super().__init__(NLI_MODEL, files, max_length, batch)
        labels = {int(k): v.lower() for k, v in self.config["id2label"].items()}
        position = {name: idx for idx, name in labels.items()}
        self.columns = [position["entailment"], position["neutral"], position["contradiction"]]

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        encodings = self.tokenizer.encode_batch([list(p) for p in pairs])
        logits = np.stack(self._run(encodings)).astype(np.float32)
        return softmax(logits)[:, self.columns]


def load_onnx_models() -> tuple[OnnxDenseEncoder, OnnxCrossEncoder, OnnxNLI]:
    """Load the three pinned models from the local Hugging Face cache (no network)."""
    from crag_verifier.fetch import model_file

    def files(model: HubModel) -> dict[str, Path]:
        return {name: model_file(model, name, offline=True) for name in model.files}

    return (
        OnnxDenseEncoder(files(DENSE_MODEL)),
        OnnxCrossEncoder(files(CROSS_ENCODER_MODEL)),
        OnnxNLI(files(NLI_MODEL)),
    )
