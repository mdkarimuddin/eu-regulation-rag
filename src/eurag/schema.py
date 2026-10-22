"""Documents and chunks, plus JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

LAW_NAMES = {"gdpr": "GDPR", "aiact": "AI Act", "test": "Test"}


@dataclass(frozen=True)
class Doc:
    """One article (or annex) of a law. ``id`` looks like ``gdpr-art-17`` or ``aiact-annex-iii``."""

    id: str
    law: str
    kind: str  # "article" or "annex"
    number: str  # "17", "4a", "III"
    title: str
    text: str
    source_url: str = ""

    @property
    def label(self) -> str:
        name = LAW_NAMES.get(self.law, self.law)
        what = "Article" if self.kind == "article" else "Annex"
        return f"{name} {what} {self.number}"

    @property
    def heading(self) -> str:
        return f"{self.label} - {self.title}" if self.title else self.label


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc id>#<n>"
    doc_id: str
    law: str
    heading: str
    text: str  # starts with the heading, so the embedder sees where the passage comes from
    index: int


def make_doc_id(law: str, kind: str, number: str) -> str:
    return f"{law}-{'art' if kind == 'article' else 'annex'}-{number.lower()}"


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {n}: invalid JSON ({exc.msg})") from exc
    return rows


def save_docs(path: str | Path, docs: list[Doc]) -> None:
    write_jsonl(path, [asdict(d) for d in docs])


def load_docs(path: str | Path) -> list[Doc]:
    docs = []
    for row in read_jsonl(path):
        try:
            docs.append(Doc(**row))
        except TypeError as exc:
            raise ValueError(f"{path}: bad document record ({exc})") from exc
    return docs
