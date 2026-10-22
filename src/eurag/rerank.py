"""Rerankers score (query, passage) pairs. A cross-encoder is slower than the retrievers but more precise."""

from __future__ import annotations

from typing import Protocol

from eurag.bm25 import tokenize

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("install the extra: pip install 'eu-regulation-rag[embeddings]'") from exc
        self.name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, p) for p in passages])]


class OverlapReranker:
    """Word-overlap scorer for tests."""

    name = "overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(p))) / (1 + len(set(tokenize(p))) ** 0.5) for p in passages]
