import re
from pathlib import Path

import pytest
from llmeval.schema import load_caseset

from eurag import config, evaltarget
from eurag.schema import save_docs

GOLD = Path(__file__).resolve().parents[1] / "eval" / "gold.yaml"
ID = re.compile(r"^(gdpr|aiact)-(art-\d+[a-z]?|annex-[ivxlcdm]+)$")


@pytest.fixture(scope="module")
def cases():
    return load_caseset(GOLD).cases


def test_gold_set_shape(cases):
    answerable = [c for c in cases if c.gold_sources]
    unanswerable = [c for c in cases if "unanswerable" in c.tags]
    assert len(answerable) >= 90 and len(unanswerable) >= 20
    assert len({c.id for c in cases}) == len(cases)


def test_gold_source_ids_are_well_formed(cases):
    for c in cases:
        for g in c.gold_sources or []:
            assert ID.match(g), f"{c.id}: {g}"


def test_gold_references_are_valid_regexes_and_questions_are_distinct(cases):
    for c in cases:
        if c.reference:
            assert c.match == "regex"
            re.compile(c.reference)
    assert len({c.input for c in cases if "injection" not in c.tags}) == len([c for c in cases if "injection" not in c.tags])


def test_unanswerable_cases_expect_refusal_and_have_no_gold(cases):
    for c in (c for c in cases if "unanswerable" in c.tags):
        assert c.expect_refusal and not c.gold_sources and not c.reference


def test_injection_case_checks_the_canary_string(cases):
    (inj,) = [c for c in cases if "injection" in c.tags]
    assert "PWNED-7731" in inj.forbidden_strings and "needs-canary-index" in inj.tags


def test_verified_cases_are_not_still_draft(cases):
    for c in cases:
        if c.gold_sources or "unanswerable" in c.tags:
            assert "draft" not in c.tags
            assert "verified-oj" in c.tags or "mine" in c.tags


def test_evaltarget_retrieve_and_ask(tmp_path, monkeypatch):
    from conftest import ScriptedLLM, doc_id, mini_docs

    processed = tmp_path / "processed"
    save_docs(processed / "t.jsonl", mini_docs())
    config.build_index(processed, config.make_embedder("hashing")).save(tmp_path / "idx")
    monkeypatch.setenv("EURAG_INDEX", str(tmp_path / "idx"))
    monkeypatch.setenv("EURAG_EMBEDDER", "hashing")
    monkeypatch.setattr(evaltarget, "_PIPELINES", {})
    monkeypatch.setattr(config, "AnthropicLLM", lambda: ScriptedLLM(lambda s, u, m: "In 30 days [test-art-1]."))
    out = evaltarget.retrieve("within how many days must an owner register a widget")
    assert out.sources[0] == doc_id("registration") and out.answer == ""
    full = evaltarget.ask("within how many days must an owner register a widget")
    assert full.cited == ["test-art-1"] and "30 days" in full.answer and full.contexts and full.tokens_in == 100
