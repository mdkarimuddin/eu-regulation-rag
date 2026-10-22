"""Vector indexes: a file-backed in-memory index (default) and an optional pgvector index.

Both expose ``chunks`` (list) and ``vector_search(query_vector, n)`` returning ``[(chunk_index, score)]``.
BM25 always runs in memory over ``chunks``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eurag.bm25 import BM25
from eurag.embed import Embedder
from eurag.schema import Chunk


class IndexError_(RuntimeError):  # noqa: N818 - avoid shadowing the builtin IndexError
    """The index on disk is missing or does not match the requested embedder."""


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


class MemoryIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict) -> None:
        if len(chunks) != len(vectors):
            raise IndexError_("chunks and vectors differ in length")
        self.chunks, self.vectors, self.meta = chunks, vectors, meta
        self.bm25 = BM25([c.text for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, params: dict | None = None, batch: int = 64) -> MemoryIndex:
        if not chunks:
            raise IndexError_("no chunks to index")
        parts = [embedder.embed([c.text for c in chunks[i : i + batch]]) for i in range(0, len(chunks), batch)]
        meta = {
            "embedder": embedder.name,
            "dim": embedder.dim,
            "fingerprint": corpus_fingerprint(chunks),
            "n_chunks": len(chunks),
            **(params or {}),
        }
        return cls(chunks, np.vstack(parts), meta)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> MemoryIndex:
        d = Path(directory)
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            vectors = np.load(d / "vectors.npy")
            chunks = [Chunk(**json.loads(ln)) for ln in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if ln]
        except FileNotFoundError as exc:
            raise IndexError_(f"no index in {d}; run: eurag build-index") from exc
        if embedder is not None and meta.get("embedder") != embedder.name:
            raise IndexError_(
                f"index was built with {meta.get('embedder')!r} but the query embedder is {embedder.name!r}"
            )
        return cls(chunks, vectors, meta)

    def vector_search(self, query_vector: np.ndarray, n: int) -> list[tuple[int, float]]:
        scores = self.vectors @ query_vector
        order = np.argsort(-scores, kind="stable")[:n]
        return [(int(i), float(scores[i])) for i in order]


class PgIndex:
    """pgvector-backed index. Needs ``pip install 'eu-regulation-rag[pg]'`` and a Postgres with the extension."""

    TABLE = "eurag_chunks"

    def __init__(self, dsn: str, dim: int) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError("install the extra: pip install 'eu-regulation-rag[pg]'") from exc
        self._psycopg = psycopg
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.dim = dim
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.TABLE} (pos int PRIMARY KEY, id text, doc_id text, law text, "
            f"heading text, body text, idx int, embedding vector({dim}))"
        )
        self.chunks = self._load_chunks()
        self.bm25 = BM25([c.text for c in self.chunks])
        self.meta = {"embedder": "pgvector", "dim": dim, "n_chunks": len(self.chunks)}

    def _load_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(f"SELECT id, doc_id, law, heading, body, idx FROM {self.TABLE} ORDER BY pos").fetchall()
        return [Chunk(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]

    def replace_all(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.conn.execute(f"TRUNCATE {self.TABLE}")
        with self.conn.cursor() as cur:
            for pos, (c, v) in enumerate(zip(chunks, vectors, strict=True)):
                cur.execute(
                    f"INSERT INTO {self.TABLE} VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (pos, c.id, c.doc_id, c.law, c.heading, c.text, c.index, v),
                )
        self.chunks = self._load_chunks()
        self.bm25 = BM25([c.text for c in self.chunks])
        self.meta["n_chunks"] = len(self.chunks)

    def vector_search(self, query_vector: np.ndarray, n: int) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            f"SELECT pos, 1 - (embedding <=> %s) AS score FROM {self.TABLE} ORDER BY embedding <=> %s LIMIT %s",
            (query_vector, query_vector, n),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]
