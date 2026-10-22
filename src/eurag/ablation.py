"""Retrieval ablations and the CI regression gate. Retrieval-only: needs no language model (except rewrite)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llmeval.schema import Case
from llmeval.stats import bootstrap_ci, mean, percentile

from eurag.chunking import chunk_corpus
from eurag.embed import Embedder
from eurag.llm import LLM
from eurag.pipeline import RagPipeline
from eurag.rerank import Reranker
from eurag.retrieve import RetrievalConfig
from eurag.schema import Doc
from eurag.store import MemoryIndex

K_VALUES = (3, 5, 10)


@dataclass
class AblationRow:
    label: str
    chunk_words: int
    n_chunks: int
    n_cases: int
    recall: dict[int, float]
    recall5_ci: tuple[float, float]
    mrr: float
    p50_ms: float
    p95_ms: float


@dataclass
class AblationResult:
    rows: list[AblationRow] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def evaluate_retrieval(pipeline: RagPipeline, cases: list[Case], chunk_words: int = 0) -> AblationRow:
    answerable = [c for c in cases if c.gold_sources]
    if not answerable:
        raise ValueError("the gold set has no answerable cases (cases with gold_sources)")
    per_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
    rr: list[float] = []
    latencies: list[float] = []
    for case in answerable:
        started = time.perf_counter()
        ranked = pipeline.retrieve(case.input).doc_ids
        latencies.append((time.perf_counter() - started) * 1000)
        gold = set(case.gold_sources or [])
        for k in K_VALUES:
            per_k[k].append(len(gold & set(ranked[:k])) / len(gold))
        rr.append(next((1 / r for r, d in enumerate(ranked, start=1) if d in gold), 0.0))
    lo, hi = bootstrap_ci(per_k[5])
    return AblationRow(
        label=pipeline.config.label(),
        chunk_words=chunk_words,
        n_chunks=len(pipeline.index.chunks),
        n_cases=len(answerable),
        recall={k: mean(v) for k, v in per_k.items()},
        recall5_ci=(lo, hi),
        mrr=mean(rr),
        p50_ms=percentile(latencies, 0.5),
        p95_ms=percentile(latencies, 0.95),
    )


def build_memory_index(docs: list[Doc], embedder: Embedder, chunk_words: int, overlap: int = 40) -> MemoryIndex:
    return MemoryIndex.build(
        chunk_corpus(docs, chunk_words, overlap), embedder, {"chunk_words": chunk_words, "overlap_words": overlap}
    )


def run_ablation(
    docs: list[Doc],
    cases: list[Case],
    embedder: Embedder,
    *,
    base_chunk_words: int = 300,
    chunk_sizes: tuple[int, ...] = (150, 300, 600),
    llm: LLM | None = None,
    reranker: Reranker | None = None,
) -> AblationResult:
    result = AblationResult()
    base_index = build_memory_index(docs, embedder, base_chunk_words)
    variants = [
        RetrievalConfig(mode="vector"),
        RetrievalConfig(mode="bm25"),
        RetrievalConfig(mode="hybrid"),
        RetrievalConfig(mode="hybrid", rerank=True),
        RetrievalConfig(mode="hybrid", rewrite=True),
    ]
    for cfg in variants:
        if cfg.rerank and reranker is None:
            result.skipped.append(f"{cfg.label()}: no reranker available")
            continue
        if cfg.rewrite and llm is None:
            result.skipped.append(f"{cfg.label()}: needs an LLM (set ANTHROPIC_API_KEY and pass --with-rewrite)")
            continue
        pipe = RagPipeline(base_index, embedder, llm=llm, reranker=reranker, config=cfg)
        result.rows.append(evaluate_retrieval(pipe, cases, base_chunk_words))
    for size in chunk_sizes:
        if size == base_chunk_words:
            continue  # already reported as the plain hybrid row
        index = build_memory_index(docs, embedder, size)
        pipe = RagPipeline(index, embedder, config=RetrievalConfig(mode="hybrid"))
        row = evaluate_retrieval(pipe, cases, size)
        row.label = f"hybrid, chunk {size}"
        result.rows.append(row)
    return result


def format_markdown(result: AblationResult) -> str:
    head = "| Variant | chunk (words) | chunks | recall@3 | recall@5 | recall@10 | MRR | p50 ms | p95 ms |"
    lines = [head, "|---|---|---|---|---|---|---|---|---|"]
    for r in result.rows:
        lines.append(
            f"| {r.label} | {r.chunk_words} | {r.n_chunks} | {r.recall[3]:.3f} | "
            f"{r.recall[5]:.3f} [{r.recall5_ci[0]:.2f}, {r.recall5_ci[1]:.2f}] | {r.recall[10]:.3f} | "
            f"{r.mrr:.3f} | {r.p50_ms:.1f} | {r.p95_ms:.1f} |"
        )
    n = result.rows[0].n_cases if result.rows else 0
    lines.append(f"\nn = {n} answerable questions. Brackets are 95% bootstrap intervals for recall@5.")
    for s in result.skipped:
        lines.append(f"\nSkipped: {s}")
    return "\n".join(lines)


def save_result(result: AblationResult, directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ablation.json").write_text(
        json.dumps({"rows": [asdict(r) for r in result.rows], "skipped": result.skipped}, indent=2), encoding="utf-8"
    )
    (d / "ablation.md").write_text(format_markdown(result) + "\n", encoding="utf-8")


def make_baseline(row: AblationRow, embedder_name: str) -> dict:
    return {
        "variant": row.label,
        "chunk_words": row.chunk_words,
        "embedder": embedder_name,
        "n_cases": row.n_cases,
        "recall_at_5": round(row.recall[5], 4),
    }


def check_gate(current_recall5: float, baseline: dict, max_drop: float = 0.03) -> tuple[bool, str]:
    """Fail when recall@5 falls by more than ``max_drop`` (absolute, so 0.03 = three points)."""
    base = float(baseline["recall_at_5"])
    drop = base - current_recall5
    ok = drop <= max_drop + 1e-9
    verb = "within" if ok else "EXCEEDS"
    return ok, f"recall@5 {current_recall5:.3f} vs baseline {base:.3f} (drop {drop:+.3f}, {verb} the {max_drop:.2f} limit)"
