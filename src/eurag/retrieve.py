"""Vector, BM25 and hybrid retrieval with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

from eurag.embed import Embedder

MODES = ("vector", "bm25", "hybrid")


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"
    per_retriever: int = 50  # chunks taken from each retriever before fusion
    candidates: int = 30  # chunks kept after fusion (this ranking decides recall@k)
    rrf_k: int = 60
    rerank: bool = False
    rewrite: bool = False
    final_chunks: int = 6  # chunks passed to the language model

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if min(self.per_retriever, self.candidates, self.final_chunks, self.rrf_k) < 1:
            raise ValueError("retrieval sizes must be positive")

    def label(self) -> str:
        parts = [self.mode]
        if self.rerank:
            parts.append("rerank")
        if self.rewrite:
            parts.append("rewrite")
        return "+".join(parts)


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank). Ties break on the lower index."""
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def search_chunks(index, embedder: Embedder, query: str, cfg: RetrievalConfig) -> list[tuple[int, float]]:
    """Return up to ``cfg.candidates`` (chunk index, score) pairs, best first."""
    lists: list[list[int]] = []
    vector_hits: list[tuple[int, float]] = []
    if cfg.mode in ("vector", "hybrid"):
        qvec = embedder.embed([query], is_query=True)[0]
        vector_hits = index.vector_search(qvec, cfg.per_retriever)
        lists.append([i for i, _ in vector_hits])
    bm25_hits: list[tuple[int, float]] = []
    if cfg.mode in ("bm25", "hybrid"):
        bm25_hits = index.bm25.search(query, cfg.per_retriever)
        lists.append([i for i, _ in bm25_hits])
    if cfg.mode == "vector":
        return vector_hits[: cfg.candidates]
    if cfg.mode == "bm25":
        return bm25_hits[: cfg.candidates]
    return rrf(lists, cfg.rrf_k)[: cfg.candidates]


def doc_ranking(chunks, ordered_chunk_indexes: list[int]) -> list[str]:
    """Collapse a chunk ranking to unique document ids, keeping first-seen order."""
    seen: dict[str, None] = {}
    for i in ordered_chunk_indexes:
        seen.setdefault(chunks[i].doc_id, None)
    return list(seen)
