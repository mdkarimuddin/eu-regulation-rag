"""Build a pipeline from an index directory and environment settings."""

from __future__ import annotations

import os
from pathlib import Path

from eurag.chunking import chunk_corpus
from eurag.embed import Embedder, make_embedder
from eurag.llm import AnthropicLLM
from eurag.pipeline import RagPipeline
from eurag.rerank import CrossEncoderReranker
from eurag.retrieve import RetrievalConfig
from eurag.schema import Doc, load_docs
from eurag.store import MemoryIndex, PgIndex

CANARY_PATH = Path(__file__).resolve().parents[2] / "data" / "canary" / "canary.jsonl"


def load_corpus(processed_dir: str | Path, with_canary: bool = False) -> list[Doc]:
    docs: list[Doc] = []
    for path in sorted(Path(processed_dir).glob("*.jsonl")):
        docs.extend(load_docs(path))
    if with_canary:
        docs.extend(load_docs(CANARY_PATH))
    if not docs:
        raise FileNotFoundError(f"no documents in {processed_dir}; run: eurag fetch --law gdpr and --law aiact")
    return docs


def build_index(
    processed_dir: str | Path,
    embedder: Embedder,
    chunk_words: int = 300,
    overlap_words: int = 40,
    with_canary: bool = False,
) -> MemoryIndex:
    chunks = chunk_corpus(load_corpus(processed_dir, with_canary), chunk_words, overlap_words)
    return MemoryIndex.build(
        chunks,
        embedder,
        {"chunk_words": chunk_words, "overlap_words": overlap_words, "with_canary": with_canary},
    )


def pipeline_from_env(with_llm: bool = True) -> RagPipeline:
    """Settings: EURAG_PG_DSN (pgvector) or EURAG_INDEX (files); EURAG_EMBEDDER (st|hashing); EURAG_MODE;
    EURAG_RERANK=1; EURAG_REWRITE=1."""
    embedder = make_embedder(os.environ.get("EURAG_EMBEDDER", "st"))
    dsn = os.environ.get("EURAG_PG_DSN")
    index = PgIndex(dsn, embedder.dim) if dsn else MemoryIndex.load(os.environ.get("EURAG_INDEX", "index"), embedder)
    cfg = RetrievalConfig(
        mode=os.environ.get("EURAG_MODE", "hybrid"),
        rerank=os.environ.get("EURAG_RERANK") == "1",
        rewrite=os.environ.get("EURAG_REWRITE") == "1",
    )
    llm = AnthropicLLM() if (with_llm or cfg.rewrite) else None
    reranker = CrossEncoderReranker() if cfg.rerank else None
    return RagPipeline(index, embedder, llm=llm, reranker=reranker, config=cfg)
