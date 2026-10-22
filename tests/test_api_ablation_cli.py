import json
from pathlib import Path

import pytest
import yaml
from conftest import ScriptedLLM, doc_id, mini_docs
from fastapi.testclient import TestClient
from llmeval.schema import Case

from eurag import ablation, cli, config
from eurag.api import add_docs, create_app
from eurag.embed import HashingEmbedder
from eurag.rerank import OverlapReranker
from eurag.schema import save_docs

# ---------- API ----------


@pytest.fixture
def client(make_pipeline):
    llm = ScriptedLLM(lambda s, u, m: "Owners register within 30 days [test-art-1].")
    pipe = make_pipeline(llm=llm)
    return TestClient(create_app(lambda: pipe)), pipe


def test_health_and_home(client):
    c, pipe = client
    assert c.get("/health").json() == {"status": "ok", "chunks": len(pipe.index.chunks), "embedder": "hashing-512"}
    assert "not legal advice" in c.get("/").text


def test_ask_endpoint(client):
    c, _ = client
    r = c.post("/ask", json={"question": "How many days to register a widget?"})
    body = r.json()
    assert r.status_code == 200 and body["cited"] == ["test-art-1"] and body["sources"][0] == doc_id("registration")
    assert set(body) >= {"answer", "abstained", "latency_ms", "tokens_in", "tokens_out"}


@pytest.mark.parametrize("question", ["", "hi", "x" * 501])
def test_ask_validates_length(client, question):
    c, _ = client
    assert c.post("/ask", json={"question": question}).status_code == 422


def test_ask_errors_do_not_leak_internals():
    def boom():
        raise RuntimeError("secret path /home/user/key")

    c = TestClient(create_app(boom))
    r = c.post("/ask", json={"question": "valid question here"})
    assert r.status_code == 503 and "secret" not in r.text and "RuntimeError" in r.text
    assert c.get("/health").status_code == 503


DOC = {"id": "test-art-20", "law": "test", "number": "20", "title": "Hats", "text": "Every owner must wear a purple hat."}


def test_ingest_is_disabled_without_a_token(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("EURAG_INGEST_TOKEN", raising=False)
    assert c.post("/ingest", json={"docs": [DOC]}, headers={"X-Ingest-Token": "x"}).status_code == 403


def test_ingest_requires_the_right_token_and_adds_documents(client, monkeypatch):
    c, pipe = client
    monkeypatch.setenv("EURAG_INGEST_TOKEN", "s3cret")
    assert c.post("/ingest", json={"docs": [DOC]}).status_code == 401
    assert c.post("/ingest", json={"docs": [DOC]}, headers={"X-Ingest-Token": "wrong"}).status_code == 401
    before = len(pipe.index.chunks)
    r = c.post("/ingest", json={"docs": [DOC]}, headers={"X-Ingest-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["chunks_total"] == before + 1
    assert pipe.retrieve("purple hat").doc_ids[0] == "test-art-20"


def test_add_docs_replaces_documents_with_the_same_id(make_pipeline):
    pipe = make_pipeline()
    n = len(pipe.index.chunks)
    from eurag.schema import Doc

    add_docs(pipe, [Doc("test-art-1", "test", "article", "1", "Duty to register", "Registration now takes 90 days.")])
    assert len(pipe.index.chunks) == n
    assert pipe.retrieve("registration 90 days").doc_ids[0] == "test-art-1"


# ---------- ablation ----------
def gold_cases():
    return [
        Case("q1", "within how many days must an owner register a widget", gold_sources=[doc_id("registration")]),
        Case("q2", "fireproof packaging for batteries", gold_sources=[doc_id("transport")]),
        Case("q3", "warranty for manufacturing defects", gold_sources=[doc_id("warranty")]),
        Case("q4", "what is the weather", expect_refusal=True),  # unanswerable: excluded from retrieval metrics
    ]


def test_evaluate_retrieval_metrics(make_pipeline):
    row = ablation.evaluate_retrieval(make_pipeline(mode="bm25"), gold_cases(), chunk_words=300)
    assert row.n_cases == 3 and row.recall[3] == 1.0 and row.mrr == 1.0 and row.label == "bm25"
    assert row.p95_ms >= row.p50_ms >= 0
    with pytest.raises(ValueError, match="no answerable"):
        ablation.evaluate_retrieval(make_pipeline(), [Case("u", "q", expect_refusal=True)])


def test_run_ablation_rows_and_skips():
    result = ablation.run_ablation(mini_docs(), gold_cases(), HashingEmbedder(), chunk_sizes=(150, 300, 600))
    labels = [r.label for r in result.rows]
    assert labels[:3] == ["vector", "bm25", "hybrid"] and "hybrid, chunk 150" in labels and "hybrid, chunk 600" in labels
    assert "hybrid, chunk 300" not in labels  # the base size is the plain hybrid row
    assert len(result.skipped) == 2 and any("rewrite" in s for s in result.skipped)
    md = ablation.format_markdown(result)
    assert md.startswith("| Variant") and "n = 3" in md and "Skipped" in md


def test_run_ablation_with_optional_components():
    llm = ScriptedLLM(lambda s, u, m: "widget registration")
    result = ablation.run_ablation(
        mini_docs(), gold_cases(), HashingEmbedder(), chunk_sizes=(300,), llm=llm, reranker=OverlapReranker()
    )
    assert {"hybrid+rerank", "hybrid+rewrite"} <= {r.label for r in result.rows} and result.skipped == []


def test_gate_logic():
    base = {"recall_at_5": 0.80}
    assert ablation.check_gate(0.79, base, 0.03)[0] is True
    assert ablation.check_gate(0.77, base, 0.03)[0] is True  # exactly three points is allowed
    ok, message = ablation.check_gate(0.76, base, 0.03)
    assert ok is False and "EXCEEDS" in message
    assert ablation.check_gate(0.95, base)[0] is True


def test_baseline_and_save(tmp_path, make_pipeline):
    row = ablation.evaluate_retrieval(make_pipeline(), gold_cases(), 300)
    b = ablation.make_baseline(row, "hashing-512")
    assert b["recall_at_5"] == round(row.recall[5], 4) and b["embedder"] == "hashing-512"
    res = ablation.AblationResult(rows=[row])
    ablation.save_result(res, tmp_path)
    assert json.loads((tmp_path / "ablation.json").read_text())["rows"][0]["label"] == "hybrid"


# ---------- CLI ----------
@pytest.fixture
def workspace(tmp_path):
    processed = tmp_path / "processed"
    save_docs(processed / "test.jsonl", mini_docs())
    gold = {
        "name": "g",
        "k": 5,
        "scorers": ["recall_at_k", "mrr"],
        "cases": [
            {"id": "q1", "input": "within how many days must an owner register a widget", "gold_sources": [doc_id("registration")]},
            {"id": "q2", "input": "fireproof packaging for batteries", "gold_sources": [doc_id("transport")]},
            {"id": "q3", "input": "what is the weather", "expect_refusal": True},
        ],
    }
    gold_path = tmp_path / "gold.yaml"
    gold_path.write_text(yaml.safe_dump(gold))
    return tmp_path, processed, gold_path


def run(*args):
    return cli.main([str(a) for a in args])


def test_cli_build_index_and_canary(workspace, capsys):
    tmp, processed, _ = workspace
    assert run("build-index", "--processed", processed, "--embedder", "hashing", "--out", tmp / "idx") == 0
    assert "indexed 10 chunks" in capsys.readouterr().out
    assert run("build-index", "--processed", processed, "--embedder", "hashing", "--with-canary", "--out", tmp / "idx2") == 0
    chunks = (tmp / "idx2" / "chunks.jsonl").read_text()
    assert "PWNED-7731" in chunks and "PWNED-7731" not in (tmp / "idx" / "chunks.jsonl").read_text()


def test_cli_check_gold(workspace, capsys):
    tmp, processed, gold = workspace
    assert run("check-gold", "--gold", gold, "--processed", processed) == 0
    assert "3 cases: 2 answerable, 1 without gold sources" in capsys.readouterr().out


def test_cli_check_gold_detects_unknown_ids(workspace, capsys):
    tmp, processed, gold = workspace
    data = yaml.safe_load(gold.read_text())
    data["cases"][0]["gold_sources"] = ["gdpr-art-9999"]
    gold.write_text(yaml.safe_dump(data))
    assert run("check-gold", "--gold", gold, "--processed", processed) == 1
    assert "gdpr-art-9999" in capsys.readouterr().out


def test_cli_ablate_and_gate(workspace, capsys):
    tmp, processed, gold = workspace
    baseline = tmp / "baseline.json"
    assert run("ablate", "--gold", gold, "--processed", processed, "--embedder", "hashing",
               "--out", tmp / "res", "--write-baseline", baseline) == 0
    out = capsys.readouterr().out
    assert "| hybrid |" in out and baseline.exists()
    assert run("gate", "--gold", gold, "--processed", processed, "--embedder", "hashing", "--baseline", baseline) == 0
    data = json.loads(baseline.read_text())
    data["recall_at_5"] = 1.5  # an unreachable baseline: the gate must fail
    baseline.write_text(json.dumps(data))
    assert run("gate", "--gold", gold, "--processed", processed, "--embedder", "hashing", "--baseline", baseline) == 1
    data["embedder"] = "other"
    baseline.write_text(json.dumps(data))
    assert run("gate", "--gold", gold, "--processed", processed, "--embedder", "hashing", "--baseline", baseline) == 2


def test_cli_errors(workspace, monkeypatch, capsys):
    tmp, processed, gold = workspace
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert run("build-index", "--processed", tmp / "empty", "--embedder", "hashing") == 2
    assert run("ask", "q?", "--index", tmp / "nothing", "--embedder", "hashing") == 2
    monkeypatch.setattr("eurag.corpus.fetch_html", lambda law: (_ for _ in ()).throw(config_error()))
    assert run("fetch", "--law", "gdpr", "--out", tmp / "d") == 2
    assert "--html" in capsys.readouterr().err
    assert run("fetch", "--law", "all", "--html", "x.html") == 2


def config_error():
    from eurag.corpus import FetchBlockedError

    return FetchBlockedError("blocked; use --html FILE")


def test_cli_fetch_from_saved_html(tmp_path, monkeypatch):
    from eurag import corpus

    monkeypatch.setitem(corpus.LAWS["gdpr"], "expected_articles", 3)
    html = Path(__file__).parent / "fixtures" / "mini_eurlex.html"
    assert run("fetch", "--law", "gdpr", "--html", html, "--out", tmp_path) == 0
    assert (tmp_path / "processed" / "gdpr.jsonl").exists() and (tmp_path / "SOURCES.md").exists()


def test_load_corpus_with_canary(workspace):
    _, processed, _ = workspace
    docs = config.load_corpus(processed, with_canary=True)
    assert docs[-1].id == "test-art-1" and "PWNED-7731" in docs[-1].text
