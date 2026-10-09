from __future__ import annotations

import hashlib
import io
import sqlite3
import tarfile
from pathlib import Path

import numpy as np
import pytest

from crag_verifier import fetch
from crag_verifier.artifacts import ALL_MODELS, DATA_DIR_ENV, RemoteFile, data_dir
from crag_verifier.cache import CachedNLI, CachedPairScorer, ScoreCache
from crag_verifier.fakes import LexicalNLI, OverlapCrossEncoder


def test_data_dir_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.delenv(DATA_DIR_ENV)
    assert data_dir() == Path.home() / ".cache" / "corrective-rag-claim-verifier"


def test_every_pinned_artifact_has_a_sha256() -> None:
    for model in ALL_MODELS:
        assert model.onnx_file in model.sha256 and "tokenizer.json" in model.sha256
        assert all(len(v) == 64 for v in model.sha256.values())
        assert len(model.revision) == 40


def test_verify_detects_tampering(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    fetch.verify(f, hashlib.sha256(b"abc").hexdigest())
    with pytest.raises(fetch.ChecksumError):
        fetch.verify(f, "0" * 64)


def _archive(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(f"data/{name}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        evil = tarfile.TarInfo("../outside.txt")
        evil.size = 1
        tar.addfile(evil, io.BytesIO(b"!"))
    return buf.getvalue()


def test_fetch_scifact_extracts_only_pinned_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    files = {"corpus.jsonl": b'{"doc_id": 1}\n', "claims_dev.jsonl": b'{"id": 1}\n'}
    blob = _archive(files)
    monkeypatch.setattr(
        fetch, "SCIFACT_FILES", {n: hashlib.sha256(c).hexdigest() for n, c in files.items()}
    )
    monkeypatch.setattr(
        fetch,
        "SCIFACT_ARCHIVE",
        RemoteFile(
            "https://example.invalid/data.tar.gz", hashlib.sha256(blob).hexdigest(), "data.tar.gz"
        ),
    )

    def fake_download(remote: RemoteFile, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        fetch.verify(dest, remote.sha256)
        return dest

    monkeypatch.setattr(fetch, "_download", fake_download)
    out = fetch.fetch_scifact(tmp_path)
    assert sorted(p.name for p in out.iterdir()) == ["claims_dev.jsonl", "corpus.jsonl"]
    assert not (tmp_path / "outside.txt").exists()
    # second call only re-verifies what is on disk
    monkeypatch.setattr(fetch, "_download", lambda *a: pytest.fail("should not download"))
    assert fetch.fetch_scifact(tmp_path) == out


def test_score_cache_computes_each_pair_once(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path / "scores.sqlite")
    ce = CachedPairScorer(OverlapCrossEncoder(), cache)
    pairs = [("insulin mice", "insulin lowered glucose in mice"), ("a b", "c d")] * 2
    first = ce.score(pairs)
    assert cache.misses == 2 and cache.hits == 2
    np.testing.assert_allclose(first, OverlapCrossEncoder().score(pairs))
    second = ce.score(pairs)
    assert cache.misses == 2 and cache.hits == 6
    np.testing.assert_allclose(first, second)
    nli = CachedNLI(LexicalNLI(), cache)
    np.testing.assert_allclose(nli.predict(pairs[:1]), LexicalNLI().predict(pairs[:1]), rtol=1e-6)
    assert ce.score([]).shape == (0,) and nli.predict([]).shape == (0, 3)
    cache.close()


def test_score_cache_discards_values_from_another_version(tmp_path: Path) -> None:
    path = tmp_path / "scores.sqlite"
    pairs = [("insulin mice", "insulin lowered glucose in mice")]
    cache = ScoreCache(path)
    CachedPairScorer(OverlapCrossEncoder(), cache).score(pairs)
    cache.close()

    reopened = ScoreCache(path)
    CachedPairScorer(OverlapCrossEncoder(), reopened).score(pairs)
    assert reopened.hits == 1 and reopened.misses == 0
    reopened.close()

    with sqlite3.connect(path) as db:
        db.execute("UPDATE meta SET version = version - 1")
    db.close()
    stale = ScoreCache(path)
    CachedPairScorer(OverlapCrossEncoder(), stale).score(pairs)
    assert stale.hits == 0 and stale.misses == 1
    stale.close()
