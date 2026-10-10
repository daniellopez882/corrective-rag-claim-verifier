"""The benchmark harness behind ``crag-verifier bench``.

One run: build or load the dense index, fit the policy on SciFact train, run the six
systems on SciFact dev, score them with the official evaluator (and the in-repo metric,
which must agree), bootstrap paired CIs, measure the evaluator's calibration, the action
mix and what abstaining gave up, and time real HTTP calls to /verify. The timing goes to
its own file so that quality numbers can be re-run without re-timing on a busy machine.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import json
import os
import platform
import random
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from crag_verifier.artifacts import ALL_MODELS, data_dir
from crag_verifier.cache import CachedNLI, CachedPairScorer, CachedQueryEncoder, ScoreCache
from crag_verifier.calibration import (
    auroc,
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    log_loss,
    reliability_bins,
)
from crag_verifier.data import Claim, Corpus, load_claims, write_jsonl
from crag_verifier.evaluator import FEATURES, gold_in_top, out_of_fold_scores
from crag_verifier.fetch import evaluator_path, scifact_dir
from crag_verifier.fit import StderrProgress, claim_utility, fit_policy
from crag_verifier.pipeline import SYSTEMS, System, Verifier
from crag_verifier.policy import Action, action_counts
from crag_verifier.reliability_svg import reliability_svg
from crag_verifier.runtime import ModelSet, build_pipeline, load_models, load_scifact_corpus
from crag_verifier.scifact_metrics import bootstrap, count_matrix, f1_scores

PACKAGES = ("numpy", "onnxruntime", "tokenizers", "fastapi", "uvicorn", "huggingface-hub")

# (system, baseline) pairs with a paired bootstrap CI. Every system against standard RAG;
# CRAG also against both strip systems without triage, which isolates what triage adds.
COMPARISONS: tuple[tuple[System, System], ...] = (
    *((s, System.STANDARD_RAG) for s in System if s is not System.STANDARD_RAG),
    (System.CRAG, System.STANDARD_RAG_STRIPS),
    (System.CRAG, System.ALWAYS_EXPAND_STRIPS),
)


@dataclass
class BenchConfig:
    out: Path
    runs_dir: Path
    train_limit: int | None = None
    dev_limit: int | None = None
    corpus_sample: int | None = None  # extra random abstracts on top of the cited ones
    latency_requests: int = 100
    bootstrap_resamples: int = 1000
    folds: int = 5
    seed: int = 0
    save_policy: bool = True
    command: str = ""


# Machine description -------------------------------------------------------------------


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gb() -> float:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        return round(status.ullTotalPhys / 1024**3, 1)
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        kb = int(meminfo.read_text().split("MemTotal:")[1].split()[0])
        return round(kb / 1024**2, 1)
    return float("nan")


def machine_info() -> dict[str, Any]:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return {
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "accelerator": "none (CPU only)",
    }


def git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


# Official evaluator --------------------------------------------------------------------


def run_official(labels: Path, predictions: Path, out: Path) -> dict[str, float]:
    script = evaluator_path()
    if not script.exists():
        raise FileNotFoundError(f"{script} missing; run `crag-verifier fetch`")
    subprocess.run(
        [
            sys.executable,
            "-I",  # isolated: the downloaded script sees only the standard library
            "-X",
            "utf8",
            str(script),
            "--labels_file",
            str(labels),
            "--preds_file",
            str(predictions),
            "--metrics_output_file",
            str(out),
            "--verbose",
        ],
        check=True,
    )
    return json.loads(out.read_text(encoding="utf-8"))


# Latency over real HTTP -----------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def measure_latency(verifier: Verifier, claims: list[Claim], warmup: int = 2) -> dict[str, Any]:
    import httpx
    import uvicorn

    from crag_verifier.api import create_app

    app = create_app(verifier, "onnx", "benchmark policy")
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    wall: list[float] = []
    steps: dict[str, list[float]] = {}
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120.0) as client:
            for claim in claims[:warmup]:
                client.post("/verify", json={"claim": claim.text}).raise_for_status()
            for claim in claims:
                start = time.perf_counter()
                resp = client.post("/verify", json={"claim": claim.text})
                wall.append((time.perf_counter() - start) * 1000.0)
                resp.raise_for_status()
                for step in resp.json()["trace"]:
                    steps.setdefault(step["step"], []).append(step["ms"])
    finally:
        server.should_exit = True
        thread.join(timeout=30)
    arr = np.array(wall)
    return {
        "requests": len(wall),
        "warmup_requests": min(warmup, len(claims)),
        "system": System.CRAG.value,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
        "step_median_ms": {k: float(np.median(v)) for k, v in steps.items()},
        "step_count": {k: len(v) for k, v in steps.items()},
        "client": "httpx, sequential, one connection, loopback",
    }


# Subsetting for smoke runs ---------------------------------------------------------------


def _subset(
    corpus: Corpus, train: list[Claim], dev: list[Claim], cfg: BenchConfig
) -> tuple[Corpus, list[Claim], list[Claim]]:
    train = train[: cfg.train_limit] if cfg.train_limit else train
    dev = dev[: cfg.dev_limit] if cfg.dev_limit else dev
    if cfg.corpus_sample is None:
        return corpus, train, dev
    keep = {d for c in train + dev for d in (*c.evidence, *c.cited_doc_ids)}
    others = sorted(d.doc_id for d in corpus if d.doc_id not in keep)
    keep |= set(random.Random(cfg.seed).sample(others, min(cfg.corpus_sample, len(others))))
    return corpus.subset(keep), train, dev


# The run ---------------------------------------------------------------------------------


def _evaluator_on_dev(
    model_x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    verifier: Verifier,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    """Raw vs Platt vs isotonic, every calibrator fitted on train out-of-fold scores only."""
    oof = out_of_fold_scores(model_x_train, y_train, k=folds, seed=seed)
    raw_dev = verifier.policy.evaluator.raw_scores(x_dev)
    out: dict[str, Any] = {"target": "a gold abstract is in the top 3 retrieved", "n": len(y_dev)}
    out["positive_rate"] = float(y_dev.mean())
    out["methods"] = {}
    for method in ("raw", "platt", "isotonic"):
        p = fit_calibrator(method, oof, y_train).predict(raw_dev)
        out["methods"][method] = {
            "auroc": auroc(p, y_dev),
            "ece_10bins": expected_calibration_error(p, y_dev),
            "brier": brier_score(p, y_dev),
            "log_loss": log_loss(p, y_dev),
            "reliability": reliability_bins(p, y_dev),
        }
    out["auroc_raw_score"] = auroc(raw_dev, y_dev)
    out["policy_calibration"] = verifier.policy.evaluator.calibrator.method
    return out


def abstained_labels(counts: np.ndarray, abstained: list[int]) -> dict[str, int]:
    """Correct and wrong abstract labels a system produced on the given claims."""
    rows = counts[abstained]
    correct = int(rows[:, 2].sum())  # abstract_correct_label_only
    return {"correct_labels": correct, "wrong_labels": int(rows[:, 1].sum()) - correct}


def run_benchmark(cfg: BenchConfig) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The results document and, when requests were timed, the latency document."""
    started = time.perf_counter()
    timings: dict[str, float] = {}
    root = data_dir()
    full_corpus = load_scifact_corpus()
    train_all = load_claims(scifact_dir() / "claims_train.jsonl")
    dev_all = load_claims(scifact_dir() / "claims_dev.jsonl")
    corpus, train, dev = _subset(full_corpus, train_all, dev_all, cfg)

    raw_models = load_models("onnx")
    cache = ScoreCache(root / "cache" / "scores.sqlite")
    dim = int(raw_models.encoder.encode_queries(["warm up"]).shape[1])
    cached = ModelSet(
        CachedQueryEncoder(raw_models.encoder, cache, dim),
        CachedPairScorer(raw_models.cross_encoder, cache),
        CachedNLI(raw_models.nli, cache),
    )
    t = time.perf_counter()
    pipeline, encode_seconds = build_pipeline(corpus, cached, root / "index")
    timings["index_build_s"] = time.perf_counter() - t
    timings["dense_encoding_s"] = encode_seconds

    progress = StderrProgress()
    t = time.perf_counter()
    policy, fit_report, train_runs = fit_policy(
        pipeline, train, folds=cfg.folds, seed=cfg.seed, progress=progress
    )
    timings["train_fit_s"] = time.perf_counter() - t
    policy.provenance["fitted_on"] = f"SciFact train, {len(train)} claims"
    if cfg.save_policy:
        policy.save(root / "policy.json")

    verifier = Verifier(pipeline, policy)
    t = time.perf_counter()
    predictions: dict[str, list[dict[str, Any]]] = {}
    actions: dict[str, list[Action]] = {}
    confidences: list[float] = []
    for system in System:
        predictions[system.value] = []
        actions[system.value] = []
        for i, claim in enumerate(dev):
            progress(f"dev:{system.value}", i, len(dev))
            result = verifier.verify(claim.text, system)
            predictions[system.value].append(result.prediction(claim.id))
            actions[system.value].append(result.action)
            if system is System.CRAG:
                confidences.append(result.confidence)
        progress(f"dev:{system.value}", len(dev), len(dev))
    timings["dev_systems_s"] = time.perf_counter() - t

    # Scoring: official script and in-repo metric on identical files.
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    labels = cfg.runs_dir / "labels_dev.jsonl"
    write_jsonl(labels, (c.raw for c in dev))
    systems: dict[str, Any] = {}
    counts: dict[str, np.ndarray] = {}
    parity = True
    for name, preds in predictions.items():
        pred_path = cfg.runs_dir / f"predictions_{name}.jsonl"
        write_jsonl(pred_path, preds)
        official = run_official(labels, pred_path, cfg.runs_dir / f"official_{name}.json")
        counts[name] = count_matrix(dev, preds)
        ours = f1_scores(counts[name].sum(axis=0))
        agree = all(abs(official[k] - ours[k]) < 1e-9 for k in official)
        parity = parity and agree
        systems[name] = {
            "description": SYSTEMS[System(name)].description,
            "official": official,
            "in_repo_metric_matches_official": agree,
            "actions": action_counts(actions[name]),
            "mean_claim_utility": float(
                np.mean([claim_utility(c, p, 1.0) for c, p in zip(dev, preds, strict=True)])
            ),
            "predicted_abstracts": int(counts[name][:, 1].sum()),
        }
    cis = bootstrap(
        counts,
        [(a.value, b.value) for a, b in COMPARISONS],
        n_resamples=cfg.bootstrap_resamples,
    )

    # Evaluator diagnostics on dev (target from the initial retrieval).
    x_dev, y_dev, y_noinfo = [], [], []
    for claim in dev:
        initial = pipeline.analyse(claim.text)
        x_dev.append(initial.features)
        y_dev.append(gold_in_top(initial.retrieval, set(claim.evidence), pipeline.top_n))
        y_noinfo.append(not claim.evidence)
    x_dev_arr, y_dev_arr = np.stack(x_dev), np.array(y_dev)
    x_train = np.stack([r.initial.features for r in train_runs])
    y_train = np.array([r.y for r in train_runs])
    evaluator_report = _evaluator_on_dev(
        x_train, y_train, x_dev_arr, y_dev_arr, verifier, cfg.folds, cfg.seed
    )

    abstention: dict[str, Any] = {}
    examples: list[dict[str, Any]] = []
    for system in (s for s in System if SYSTEMS[s].triage):
        name = system.value
        abstained = [i for i, a in enumerate(actions[name]) if a is Action.INCORRECT]
        n = len(abstained)
        abstention[name] = {
            "abstained": n,
            "rate": n / len(dev) if dev else 0.0,
            "precision_gold_noinfo": (sum(y_noinfo[i] for i in abstained) / n) if n else None,
            "precision_retrieval_wrong": (sum(1 - y_dev[i] for i in abstained) / n) if n else None,
            "gold_noinfo_claims": int(sum(y_noinfo)),
            "retrieval_wrong_claims": int(len(dev) - sum(y_dev)),
            "labels_on_abstained_claims_without_triage": {
                other.value: abstained_labels(counts[other.value], abstained)
                for other in System
                if not SYSTEMS[other].triage and SYSTEMS[other].evidence is SYSTEMS[system].evidence
            },
        }
    for i, a in enumerate(actions[System.CRAG.value]):
        if a is Action.INCORRECT and len(examples) < 6:
            examples.append(
                {
                    "claim_id": dev[i].id,
                    "claim": dev[i].text,
                    "confidence": confidences[i],
                    "gold_noinfo": bool(y_noinfo[i]),
                    "gold_in_top3": bool(y_dev[i]),
                    "features": dict(zip(FEATURES, x_dev[i].tolist(), strict=True)),
                }
            )
    abstention["examples"] = examples
    write_jsonl(
        cfg.runs_dir / "dev_claims.jsonl",
        (
            {
                "id": claim.id,
                "gold_noinfo": bool(y_noinfo[i]),
                "gold_in_top3": bool(y_dev[i]),
                "crag_confidence": confidences[i],
                "actions": {name: actions[name][i].value for name in actions},
                "predicted": {name: predictions[name][i]["evidence"] for name in predictions},
            }
            for i, claim in enumerate(dev)
        ),
    )

    latency: dict[str, Any] | None = None
    if cfg.latency_requests > 0:
        live = Verifier(
            build_pipeline(corpus, raw_models, root / "index")[0], policy
        )  # uncached models: latency includes every forward pass
        t = time.perf_counter()
        latency = measure_latency(live, dev[: cfg.latency_requests])
        timings["latency_s"] = time.perf_counter() - t
    cache.close()
    timings["total_s"] = time.perf_counter() - started
    provenance = {
        "date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        "git_sha": git_sha(),
        "command": cfg.command,
        "machine": machine_info(),
    }

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    reliability_path = cfg.out.with_name(cfg.out.stem + "-reliability.svg")
    reliability_path.write_text(
        reliability_svg(
            evaluator_report["methods"], title="SciFact dev: P(gold abstract in top 3)"
        ),
        encoding="utf-8",
    )

    result = {
        "benchmark": "SciFact dev, official claim-verification metrics",
        **provenance,
        "models": {
            m.key: {
                "repo": m.repo_id,
                "revision": m.revision,
                "file": m.onnx_file,
                "licence": m.licence,
            }
            for m in ALL_MODELS
        },
        "data": {
            "dataset": "SciFact (AllenAI), licence CC BY-NC 2.0",
            "corpus_documents": len(corpus),
            "train_claims": len(train),
            "dev_claims": len(dev),
            "subset": cfg.train_limit is not None
            or cfg.dev_limit is not None
            or cfg.corpus_sample is not None,
        },
        "config": {
            "top_n": pipeline.top_n,
            "first_stage_k": pipeline.retriever.first_stage_k,
            "rerank_depth": pipeline.retriever.rerank_depth,
            "expand_depth": pipeline.retriever.expand_depth,
            "rrf_k": pipeline.retriever.rrf_k,
            "bm25_k1": pipeline.retriever.bm25.k1,
            "bm25_b": pipeline.retriever.bm25.b,
            "folds": cfg.folds,
            "seed": cfg.seed,
            "bootstrap_resamples": cfg.bootstrap_resamples,
            "ci_level_percent": 95,
            "utility": "per claim: correct labelled abstracts minus wrong ones; abstain = 0",
        },
        "timings_s": timings,
        "train_fit": fit_report,
        "policy": policy.to_dict(),
        "dev": {
            "systems": systems,
            "metric_parity_with_official": parity,
            "bootstrap": cis,
            "evaluator": evaluator_report,
            "abstention": abstention,
            "latency": (
                f"see {latency_path(cfg.out).name}"
                if latency is not None
                else "not measured in this run (--latency-requests 0)"
            ),
        },
        "not_run": {
            "llm_query_rewrite": "not run: needs an LLM endpoint; keyphrase rewriting is used",
            "llm_explanation": "not run: needs an LLM endpoint; explanation is optional in /verify",
        },
        "artifacts": {
            "predictions_dir": cfg.runs_dir.as_posix(),
            "reliability_svg": reliability_path.name,
        },
    }
    if latency is None:
        return result, None
    return result, {
        "benchmark": "POST /verify latency over loopback HTTP, SciFact corpus, ONNX models",
        **provenance,
        "policy_thresholds": policy.to_dict()["thresholds"],
        "latency": latency,
    }


def _clean(value: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, NaN (undefined metric) to null."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def write_results(result: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(result), indent=2, allow_nan=False), encoding="utf-8")


def latency_path(out: Path) -> Path:
    return out.with_name(out.stem + "-latency.json")


def default_runs_dir() -> Path:
    """Predictions and official metric files; ``bench/runs`` is git-ignored."""
    return Path("bench") / "runs"
