"""Smoke checks for a running demo-backend container; standard library only.

Usage: python3 .github/container_smoke.py http://localhost:8000

Each fitted action path is exercised once: a claim the demo corpus supports (refine),
a claim it knows nothing about (abstain), and the always-expand system (re-retrieve).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def call(base: str, path: str, body: bytes | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path,
        data=body,
        headers={"content-type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def verify(base: str, payload: dict) -> tuple[int, dict]:
    return call(base, "/verify", json.dumps(payload).encode())


def main(base: str) -> None:
    status, health = call(base, "/health")
    assert status == 200 and health["backend"] == "demo", health

    status, supported = verify(base, {"claim": "Strain K4 degrades polyethylene film."})
    assert status == 200, supported
    assert supported["action"] == "correct" and supported["verdict"] == "SUPPORT", supported
    assert supported["evidence"][0]["rationale_sentence_ids"], supported

    status, unrelated = verify(base, {"claim": "Saffron extract improves memory in children."})
    assert status == 200, unrelated
    assert unrelated["abstained"] and unrelated["verdict"] == "NOINFO", unrelated
    assert unrelated["evidence"] == [], unrelated
    assert unrelated["trace"][-1]["step"] == "abstain", unrelated

    status, expanded = verify(
        base, {"claim": "Strain K4 degrades polyethylene film.", "system": "always_expand"}
    )
    assert status == 200 and expanded["action"] == "ambiguous", expanded
    assert "re-retrieve" in [step["step"] for step in expanded["trace"]], expanded

    assert verify(base, {"claim": "   "})[0] == 422
    assert verify(base, {"claim": "x" * 9000})[0] == 413

    for name, body in (("refine", supported), ("abstain", unrelated), ("expand", expanded)):
        print(f"{name}: {body['verdict']} action={body['action']} p={body['evaluator_confidence']}")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
