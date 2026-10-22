"""Targets for the llmeval harness.

    llmeval run eval/gold.yaml --target eurag.evaltarget:ask       # full answers (needs ANTHROPIC_API_KEY)
    llmeval run eval/gold.yaml --target eurag.evaltarget:retrieve --scorers recall_at_k mrr   # retrieval only
"""

from __future__ import annotations

from llmeval.targets import Output

from eurag.config import pipeline_from_env

_PIPELINES: dict[bool, object] = {}


def _get(with_llm: bool):
    if with_llm not in _PIPELINES:
        _PIPELINES[with_llm] = pipeline_from_env(with_llm=with_llm)
    return _PIPELINES[with_llm]


def ask(question: str) -> Output:
    a = _get(True).ask(question)
    return Output(
        answer=a.answer,
        sources=a.sources,
        contexts=a.contexts,
        cited=a.cited,
        tokens_in=a.tokens_in,
        tokens_out=a.tokens_out,
    )


def retrieve(question: str) -> Output:
    p = _get(False)
    r = p.retrieve(question)
    return Output(answer="", sources=r.doc_ids, tokens_in=r.tokens_in, tokens_out=r.tokens_out)
