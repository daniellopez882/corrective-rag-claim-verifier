"""SQLite memo for model outputs, used by the benchmark only.

The four benchmarked systems and the train-time fitting ask the same models about the
same (claim, text) pairs many times; caching makes each pair cost one forward pass.
The service never uses it, so its latency numbers include every model call.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from crag_verifier.models import DenseEncoder, NLIModel, PairScorer

_CHUNK = 400

# Bumped whenever stored values stop matching what a fresh forward pass returns.
# 2: the quantised models run one sequence at a time; version-1 values were batch-dependent.
CACHE_VERSION = 2


class ScoreCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS meta (version INTEGER)")
            row = self._db.execute("SELECT version FROM meta").fetchone()
            if row is None or row[0] != CACHE_VERSION:
                self._db.execute("DROP TABLE IF EXISTS scores")
                self._db.execute("DELETE FROM meta")
                self._db.execute("INSERT INTO meta VALUES (?)", (CACHE_VERSION,))
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS scores (model TEXT, key BLOB, value BLOB, "
                "PRIMARY KEY (model, key))"
            )
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(parts: Sequence[str]) -> bytes:
        return hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=16).digest()

    def lookup(
        self,
        model: str,
        items: Sequence[Sequence[str]],
        compute: Callable[[list[Sequence[str]]], np.ndarray],
        width: int,
    ) -> np.ndarray:
        """Rows for ``items`` (float32, ``width`` columns), computing only the misses."""
        keys = [self.key(it) for it in items]
        found: dict[bytes, np.ndarray] = {}
        unique = list(dict.fromkeys(keys))
        for start in range(0, len(unique), _CHUNK):
            chunk = unique[start : start + _CHUNK]
            marks = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT key, value FROM scores WHERE model = ? AND key IN ({marks})",
                [model, *chunk],
            ).fetchall()
            for k, v in rows:
                found[k] = np.frombuffer(v, dtype=np.float32)
        missing = [i for i, k in enumerate(keys) if k not in found]
        first_missing: dict[bytes, int] = {}
        for i in missing:
            first_missing.setdefault(keys[i], i)
        # hits: items served without a forward pass; misses: forward passes run
        self.hits += len(items) - len(first_missing)
        self.misses += len(first_missing)
        if first_missing:
            todo = list(first_missing.values())
            values = np.asarray(compute([items[i] for i in todo]), dtype=np.float32)
            values = values.reshape(len(todo), width)
            with self._db:
                self._db.executemany(
                    "INSERT OR REPLACE INTO scores VALUES (?, ?, ?)",
                    [(model, keys[i], values[j].tobytes()) for j, i in enumerate(todo)],
                )
            for j, i in enumerate(todo):
                found[keys[i]] = values[j]
        out = np.zeros((len(items), width), dtype=np.float32)
        for i, k in enumerate(keys):
            out[i] = found[k]
        return out

    def close(self) -> None:
        self._db.close()


class CachedPairScorer:
    def __init__(self, inner: PairScorer, cache: ScoreCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        rows = self.cache.lookup(self.name, pairs, lambda todo: self.inner.score(todo), 1)
        return rows[:, 0]


class CachedNLI:
    def __init__(self, inner: NLIModel, cache: ScoreCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        return self.cache.lookup(self.name, pairs, lambda todo: self.inner.predict(todo), 3)


class CachedQueryEncoder:
    """Caches query embeddings; document encoding goes through the dense index instead."""

    def __init__(self, inner: DenseEncoder, cache: ScoreCache, dim: int):
        self.inner = inner
        self.cache = cache
        self.dim = dim
        self.name = inner.name

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        items = [(t,) for t in texts]
        return self.cache.lookup(
            self.name + ":query",
            items,
            lambda todo: self.inner.encode_queries([t[0] for t in todo]),
            self.dim,
        )

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.inner.encode_documents(texts)
