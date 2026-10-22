"""eurag command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llmeval.schema import TestSetError, load_caseset

from eurag import ablation, config, corpus
from eurag.embed import make_embedder
from eurag.llm import AnthropicLLM, LLMError
from eurag.pipeline import RagPipeline
from eurag.rerank import CrossEncoderReranker
from eurag.retrieve import RetrievalConfig
from eurag.store import IndexError_, MemoryIndex, PgIndex


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eurag", description="RAG over the EU AI Act and GDPR.")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download a law from EUR-Lex (or parse a saved HTML file) into data/processed")
    f.add_argument("--law", choices=[*corpus.LAWS, "all"], default="all")
    f.add_argument("--html", help="parse this saved HTML file instead of downloading (use with a single --law)")
    f.add_argument("--out", default="data")

    b = sub.add_parser("build-index", help="chunk and embed the processed corpus")
    b.add_argument("--processed", default="data/processed")
    b.add_argument("--embedder", default="st", help="st (bge-small), hashing (tests only), or a model name")
    b.add_argument("--chunk-words", type=int, default=300)
    b.add_argument("--overlap", type=int, default=40)
    b.add_argument("--with-canary", action="store_true", help="add the prompt-injection test document")
    b.add_argument("--out", default="index")
    b.add_argument("--pg-dsn", help="also load the index into Postgres/pgvector (needs the pg extra)")

    a = sub.add_parser("ask", help="answer one question from the command line")
    a.add_argument("question")
    a.add_argument("--index", default="index")
    a.add_argument("--embedder", default="st")
    a.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    a.add_argument("--rerank", action="store_true")
    a.add_argument("--rewrite", action="store_true")

    s = sub.add_parser("serve", help="run the API (needs EURAG_INDEX, EURAG_EMBEDDER, ANTHROPIC_API_KEY)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    for name, help_ in (("ablate", "run the retrieval ablation table"), ("gate", "fail if recall@5 regressed")):
        e = sub.add_parser(name, help=help_)
        e.add_argument("--gold", default="eval/gold.yaml")
        e.add_argument("--processed", default="data/processed")
        e.add_argument("--embedder", default="st")
        if name == "ablate":
            e.add_argument("--out", default="results")
            e.add_argument("--write-baseline", help="write the plain hybrid row to this baseline JSON file")
            e.add_argument("--with-rerank", action="store_true", help="include the cross-encoder variant")
            e.add_argument("--with-rewrite", action="store_true", help="include the LLM query-rewrite variant")
        else:
            e.add_argument("--baseline", default="eval/baseline.json")
            e.add_argument("--max-drop", type=float, default=0.03)

    c = sub.add_parser("check-gold", help="check that every gold source id exists in the corpus")
    c.add_argument("--gold", default="eval/gold.yaml")
    c.add_argument("--processed", default="data/processed")
    return p


def _cmd_fetch(args: argparse.Namespace) -> int:
    laws = list(corpus.LAWS) if args.law == "all" else [args.law]
    if args.html and len(laws) != 1:
        raise ValueError("--html needs a single --law")
    counts = {}
    for law in laws:
        docs = corpus.build_corpus(law, args.out, args.html)
        counts[law] = len(docs)
        print(f"{law}: {sum(d.kind == 'article' for d in docs)} articles, {sum(d.kind == 'annex' for d in docs)} annexes")
    corpus.write_sources(args.out, laws, counts)
    return 0


def _cmd_build_index(args: argparse.Namespace) -> int:
    index = config.build_index(args.processed, make_embedder(args.embedder), args.chunk_words, args.overlap, args.with_canary)
    index.save(args.out)
    if args.pg_dsn:
        PgIndex(args.pg_dsn, index.meta["dim"]).replace_all(index.chunks, index.vectors)
        print(f"loaded {len(index.chunks)} chunks into pgvector")
    print(f"indexed {len(index.chunks)} chunks with {index.meta['embedder']} into {args.out}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    embedder = make_embedder(args.embedder)
    cfg = RetrievalConfig(mode=args.mode, rerank=args.rerank, rewrite=args.rewrite)
    pipe = RagPipeline(
        MemoryIndex.load(args.index, embedder),
        embedder,
        llm=AnthropicLLM(),
        reranker=CrossEncoderReranker() if args.rerank else None,
        config=cfg,
    )
    a = pipe.ask(args.question)
    print(a.answer)
    print(f"\ncited: {', '.join(a.cited) or '-'} | top sources: {', '.join(a.sources[:5])}")
    print(f"tokens in/out: {a.tokens_in}/{a.tokens_out} | {a.latency_ms:.0f} ms")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("eurag.api:app_from_env", factory=True, host=args.host, port=args.port)
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    cases = load_caseset(args.gold).cases
    docs = config.load_corpus(args.processed)
    embedder = make_embedder(args.embedder)
    llm = AnthropicLLM() if args.with_rewrite else None
    reranker = CrossEncoderReranker() if args.with_rerank else None
    result = ablation.run_ablation(docs, cases, embedder, llm=llm, reranker=reranker)
    ablation.save_result(result, args.out)
    print(ablation.format_markdown(result))
    if args.write_baseline:
        base = next(r for r in result.rows if r.label == "hybrid" and r.chunk_words == 300)
        Path(args.write_baseline).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_baseline).write_text(json.dumps(ablation.make_baseline(base, embedder.name), indent=2), encoding="utf-8")
        print(f"\nwrote baseline to {args.write_baseline}")
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    embedder = make_embedder(args.embedder)
    if baseline.get("embedder") != embedder.name:
        raise ValueError(f"baseline used {baseline.get('embedder')!r} but --embedder is {embedder.name!r}")
    docs = config.load_corpus(args.processed)
    cases = load_caseset(args.gold).cases
    index = ablation.build_memory_index(docs, embedder, int(baseline["chunk_words"]))
    row = ablation.evaluate_retrieval(RagPipeline(index, embedder, config=RetrievalConfig(mode="hybrid")), cases)
    ok, message = ablation.check_gate(row.recall[5], baseline, args.max_drop)
    print(message)
    return 0 if ok else 1


def _cmd_check_gold(args: argparse.Namespace) -> int:
    cases = load_caseset(args.gold).cases
    ids = {d.id for d in config.load_corpus(args.processed)}
    bad = [(c.id, g) for c in cases for g in (c.gold_sources or []) if g not in ids]
    answerable = sum(1 for c in cases if c.gold_sources)
    print(f"{len(cases)} cases: {answerable} answerable, {len(cases) - answerable} without gold sources")
    for case_id, gold in bad:
        print(f"  unknown gold source {gold!r} in case {case_id}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "fetch": _cmd_fetch, "build-index": _cmd_build_index, "ask": _cmd_ask, "serve": _cmd_serve,
        "ablate": _cmd_ablate, "gate": _cmd_gate, "check-gold": _cmd_check_gold,
    }
    try:
        return handlers[args.command](args)
    except (corpus.FetchBlockedError, IndexError_, LLMError, TestSetError, FileNotFoundError, ValueError,
            RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
