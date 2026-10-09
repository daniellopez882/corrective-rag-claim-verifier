"""SciFact corpus and claim records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Label = Literal["SUPPORT", "CONTRADICT"]


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return self.title + " " + " ".join(self.sentences)


@dataclass(frozen=True)
class GoldEvidence:
    label: Label
    rationales: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Claim:
    id: int
    text: str
    evidence: dict[int, GoldEvidence] = field(default_factory=dict)
    cited_doc_ids: tuple[int, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


class Corpus:
    def __init__(self, documents: Iterable[Document]):
        self.documents: list[Document] = list(documents)
        self.position: dict[int, int] = {d.doc_id: i for i, d in enumerate(self.documents)}
        if len(self.position) != len(self.documents):
            raise ValueError("duplicate doc_id in corpus")

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def __getitem__(self, doc_id: int) -> Document:
        return self.documents[self.position[doc_id]]

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self.position

    def subset(self, doc_ids: Iterable[int]) -> Corpus:
        keep = set(doc_ids)
        return Corpus(d for d in self.documents if d.doc_id in keep)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_corpus(path: Path) -> Corpus:
    return Corpus(
        Document(int(r["doc_id"]), r["title"], tuple(r["abstract"])) for r in _read_jsonl(path)
    )


def parse_claim(record: dict[str, Any]) -> Claim:
    evidence: dict[int, GoldEvidence] = {}
    for doc_id, entries in (record.get("evidence") or {}).items():
        labels = {e["label"] for e in entries}
        if len(labels) != 1:
            raise ValueError(f"claim {record['id']}: conflicting labels for doc {doc_id}")
        evidence[int(doc_id)] = GoldEvidence(
            label=labels.pop(), rationales=tuple(tuple(e["sentences"]) for e in entries)
        )
    return Claim(
        id=int(record["id"]),
        text=record["claim"],
        evidence=evidence,
        cited_doc_ids=tuple(int(d) for d in record.get("cited_doc_ids", [])),
        raw=record,
    )


def load_claims(path: Path) -> list[Claim]:
    return [parse_claim(r) for r in _read_jsonl(path)]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
