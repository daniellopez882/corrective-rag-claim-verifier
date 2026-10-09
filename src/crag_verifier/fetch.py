"""Download SciFact, the official evaluator and the ONNX models, verifying every sha256."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx

from crag_verifier.artifacts import (
    ALL_MODELS,
    OFFICIAL_EVALUATOR,
    SCIFACT_ARCHIVE,
    SCIFACT_FILES,
    HubModel,
    RemoteFile,
    data_dir,
)


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumError(f"{path}: sha256 {actual} does not match pinned {expected}")


def _download(remote: RemoteFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == remote.sha256:
        return dest
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        with httpx.stream("GET", remote.url, follow_redirects=True, timeout=60.0) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(1 << 16):
                tmp.write(chunk)
    try:
        verify(tmp_path, remote.sha256)
    except ChecksumError:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(dest)
    return dest


def scifact_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "scifact"


def evaluator_path(root: Path | None = None) -> Path:
    return (root or data_dir()) / "evaluator" / OFFICIAL_EVALUATOR.filename


def fetch_scifact(root: Path | None = None) -> Path:
    target = scifact_dir(root)
    if all((target / n).exists() for n in SCIFACT_FILES):
        for name, digest in SCIFACT_FILES.items():
            verify(target / name, digest)
        return target
    archive = _download(SCIFACT_ARCHIVE, target / SCIFACT_ARCHIVE.filename)
    with tarfile.open(archive, "r:gz") as tar:
        for name, digest in SCIFACT_FILES.items():
            member = tar.getmember(f"data/{name}")
            src = tar.extractfile(member)
            if src is None:
                raise ChecksumError(f"{name} missing from {archive}")
            out = target / name
            with src, out.open("wb") as fh:
                shutil.copyfileobj(src, fh)
            verify(out, digest)
    archive.unlink()
    return target


def fetch_evaluator(root: Path | None = None) -> Path:
    return _download(OFFICIAL_EVALUATOR, evaluator_path(root))


def model_file(model: HubModel, filename: str, *, offline: bool = True) -> Path:
    """Resolve one model file in the Hugging Face cache (HF_HOME), downloading if allowed."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=model.repo_id,
            filename=filename,
            revision=model.revision,
            local_files_only=offline,
        )
    )
    return path


def fetch_model(model: HubModel) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for filename, digest in model.sha256.items():
        path = model_file(model, filename, offline=False)
        verify(path, digest)
        paths[filename] = path
    return paths


def fetch_all(root: Path | None = None, *, models: bool = True) -> dict[str, str]:
    report = {
        "scifact": str(fetch_scifact(root)),
        "evaluator": str(fetch_evaluator(root)),
    }
    if models:
        for model in ALL_MODELS:
            paths = fetch_model(model)
            report[model.key] = str(paths[model.onnx_file])
    return report
