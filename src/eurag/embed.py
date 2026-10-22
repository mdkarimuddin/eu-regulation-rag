"""Embedders. ``HashingEmbedder`` is for tests and CI only; real runs use a sentence-transformers model."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray: ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing into a fixed-size signed vector. Lexical, not semantic."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                mat[row, idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        return _normalise(mat)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install the extra: pip install 'eu-regulation-rag[embeddings]'") from exc
        self.model_name = model_name
        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.max_tokens = int(getattr(self._model, "max_seq_length", 512))

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if is_query and "bge" in self.model_name.lower():
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        vecs = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(DEFAULT_MODEL if name in ("st", "default") else name)
