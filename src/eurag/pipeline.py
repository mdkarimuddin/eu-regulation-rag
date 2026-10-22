"""The RAG pipeline: (rewrite) -> retrieve -> (rerank) -> answer with citations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from eurag.answer import build_prompt, parse_answer
from eurag.embed import Embedder
from eurag.llm import LLM
from eurag.rerank import Reranker
from eurag.retrieve import RetrievalConfig, doc_ranking, search_chunks

REWRITE_SYSTEM = (
    "Rewrite the user's question as a short search query that uses the vocabulary of EU legislation "
    "(for example 'data subject', 'controller', 'provider', 'high-risk AI system'). "
    "Output only the query, with no explanation."
)


@dataclass
class Retrieval:
    chunk_indexes: list[int]  # final ordering of candidate chunks
    doc_ids: list[str]  # unique article/annex ids in rank order (used for recall@k)
    rewritten: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class Answer:
    question: str
    answer: str = ""
    abstained: bool = False
    cited: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # ranked doc ids
    context_ids: list[str] = field(default_factory=list)  # passages given to the model
    contexts: list[str] = field(default_factory=list)
    rewritten: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


class RagPipeline:
    def __init__(
        self,
        index,
        embedder: Embedder,
        *,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.index, self.embedder, self.llm, self.reranker = index, embedder, llm, reranker
        self.config = config or RetrievalConfig()
        if self.config.rerank and reranker is None:
            raise ValueError("config.rerank is set but no reranker was given")
        if self.config.rewrite and llm is None:
            raise ValueError("config.rewrite is set but no llm was given")

    def retrieve(self, question: str) -> Retrieval:
        query, tin, tout, rewritten = question, 0, 0, None
        if self.config.rewrite:
            assert self.llm is not None
            reply = self.llm.complete(REWRITE_SYSTEM, question, max_tokens=60)
            tin, tout = reply.tokens_in, reply.tokens_out
            candidate = reply.text.strip().splitlines()[0].strip() if reply.text.strip() else ""
            if candidate:  # an empty rewrite falls back to the original question
                query, rewritten = candidate, candidate
        hits = search_chunks(self.index, self.embedder, query, self.config)
        order = [i for i, _ in hits]
        if self.config.rerank and order:
            assert self.reranker is not None
            scores = self.reranker.score(question, [self.index.chunks[i].text for i in order])
            order = [i for _, i in sorted(zip(scores, order, strict=True), key=lambda x: (-x[0], x[1]))]
        return Retrieval(order, doc_ranking(self.index.chunks, order), rewritten, tin, tout)

    def _passages(self, order: list[int]) -> list[tuple[str, str]]:
        merged: dict[str, list[str]] = {}
        for i in order[: self.config.final_chunks]:
            chunk = self.index.chunks[i]
            merged.setdefault(chunk.doc_id, []).append(chunk.text)
        return [(pid, "\n\n".join(texts)) for pid, texts in merged.items()]

    def ask(self, question: str) -> Answer:
        if self.llm is None:
            raise ValueError("ask() needs an llm; use retrieve() for retrieval-only runs")
        started = time.perf_counter()
        r = self.retrieve(question)
        passages = self._passages(r.chunk_indexes)
        system, user = build_prompt(question, passages)
        reply = self.llm.complete(system, user)
        text, abstained, cited, invalid = parse_answer(reply.text, [pid for pid, _ in passages])
        return Answer(
            question=question,
            answer=text,
            abstained=abstained,
            cited=cited,
            invalid_citations=invalid,
            sources=r.doc_ids,
            context_ids=[pid for pid, _ in passages],
            contexts=[t for _, t in passages],
            rewritten=r.rewritten,
            tokens_in=r.tokens_in + reply.tokens_in,
            tokens_out=r.tokens_out + reply.tokens_out,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
