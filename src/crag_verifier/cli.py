"""Command line entry point: fetch, index, fit, verify, serve, bench."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from crag_verifier.artifacts import data_dir


def _cmd_fetch(args: argparse.Namespace) -> int:
    from crag_verifier.fetch import fetch_all

    print(json.dumps(fetch_all(models=not args.no_models), indent=2))
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from crag_verifier.runtime import (
        dense_index_path,
        dense_vectors,
        load_models,
        load_scifact_corpus,
    )

    corpus = load_scifact_corpus()
    models = load_models("onnx")
    index_dir = data_dir() / "index"
    _, seconds = dense_vectors(corpus, models.encoder, index_dir)
    path = dense_index_path(corpus, models.encoder, index_dir)
    print(json.dumps({"index": str(path), "documents": len(corpus), "seconds": round(seconds, 1)}))
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    from crag_verifier.cache import CachedNLI, CachedPairScorer, CachedQueryEncoder, ScoreCache
    from crag_verifier.data import load_claims
    from crag_verifier.fetch import scifact_dir
    from crag_verifier.fit import StderrProgress, fit_policy
    from crag_verifier.runtime import ModelSet, build_pipeline, load_models, load_scifact_corpus

    raw = load_models("onnx")
    cache = ScoreCache(data_dir() / "cache" / "scores.sqlite")
    dim = int(raw.encoder.encode_queries(["warm up"]).shape[1])
    models = ModelSet(
        CachedQueryEncoder(raw.encoder, cache, dim),
        CachedPairScorer(raw.cross_encoder, cache),
        CachedNLI(raw.nli, cache),
    )
    pipeline, _ = build_pipeline(load_scifact_corpus(), models, data_dir() / "index")
    claims = load_claims(scifact_dir() / "claims_train.jsonl")
    policy, report, _ = fit_policy(pipeline, claims, progress=StderrProgress())
    policy.provenance["fitted_on"] = f"SciFact train, {len(claims)} claims"
    out = Path(args.out) if args.out else data_dir() / "policy.json"
    policy.save(out)
    cache.close()
    print(json.dumps({"policy": str(out), "report": report}, indent=2, default=str))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from crag_verifier.api import to_response
    from crag_verifier.pipeline import Explainer, System, Verifier
    from crag_verifier.providers import provider_from_env
    from crag_verifier.runtime import service_pipeline

    pipeline, policy, _ = service_pipeline(args.backend)
    provider = provider_from_env()
    verifier = Verifier(pipeline, policy, Explainer(provider) if provider else None)
    result = verifier.verify(args.claim, System(args.system), explain=args.explain)
    print(to_response(result, verifier, args.explain).model_dump_json(indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ["CRAG_VERIFIER_BACKEND"] = args.backend
    uvicorn.run(
        "crag_verifier.api:app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from crag_verifier.bench import BenchConfig, default_runs_dir, run_benchmark, write_results

    if args.smoke:
        args.train_limit = args.train_limit or 30
        args.dev_limit = args.dev_limit or 15
        args.corpus_sample = args.corpus_sample if args.corpus_sample is not None else 150
        args.latency_requests = min(args.latency_requests, 5)
        args.bootstrap = min(args.bootstrap, 200)
    out = Path(args.out)
    cfg = BenchConfig(
        out=out,
        runs_dir=Path(args.runs_dir) if args.runs_dir else default_runs_dir() / out.stem,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        corpus_sample=args.corpus_sample,
        latency_requests=args.latency_requests,
        bootstrap_resamples=args.bootstrap,
        save_policy=not args.smoke,
        command="uv run crag-verifier " + shlex.join(sys.argv[1:]),
    )
    result = run_benchmark(cfg)
    write_results(result, out)
    summary = {
        name: {k: round(v, 4) for k, v in s["official"].items() if k.endswith("_f1")}
        for name, s in result["dev"]["systems"].items()
    }
    print(json.dumps({"results": str(out), "dev_f1": summary}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crag-verifier", description="Corrective RAG claim verification on SciFact"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download SciFact, the official evaluator and the models")
    p.add_argument("--no-models", action="store_true", help="data and evaluator only")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("index", help="encode the SciFact corpus with bge-small (cached)")
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("fit", help="fit the evaluator, thresholds and aggregator on train")
    p.add_argument("--out", help="policy path (default: $CRAG_VERIFIER_DATA_DIR/policy.json)")
    p.set_defaults(func=_cmd_fit)

    p = sub.add_parser("verify", help="verify one claim and print the /verify response")
    p.add_argument("claim")
    p.add_argument(
        "--system",
        default="crag",
        choices=["standard_rag", "always_expand", "crag_no_strips", "crag"],
    )
    p.add_argument("--backend", default="onnx", choices=["onnx", "demo"])
    p.add_argument("--explain", action="store_true")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("serve", help="run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--backend",
        default=os.environ.get("CRAG_VERIFIER_BACKEND", "onnx"),
        choices=["onnx", "demo"],
    )
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("bench", help="run the full SciFact benchmark and write a results JSON")
    p.add_argument("--out", default="bench/results/scifact-dev.json")
    p.add_argument("--runs-dir", help="where predictions and official metric files go")
    p.add_argument("--smoke", action="store_true", help="tiny subset; proves the harness runs")
    p.add_argument("--train-limit", type=int)
    p.add_argument("--dev-limit", type=int)
    p.add_argument("--corpus-sample", type=int, help="random extra abstracts beside cited ones")
    p.add_argument("--latency-requests", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.set_defaults(func=_cmd_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
