import urllib.error
from pathlib import Path

import pytest

from eurag import corpus
from eurag.schema import load_docs

FIXTURE = Path(__file__).parent / "fixtures" / "mini_eurlex.html"


@pytest.fixture
def lines():
    return corpus.html_to_lines(FIXTURE.read_text(encoding="utf-8"))


def test_html_to_lines_drops_style_and_merges_list_markers(lines):
    joined = "\n".join(lines)
    assert "color:red" not in joined and "synthetic fixture" not in joined
    assert any(ln.startswith("(a) ") and "small device" in ln for ln in lines)
    assert "Owners shall register each widget within 30 days. Late registration is an infringement." in joined


def test_split_documents_ids_titles_and_text(lines):
    docs = {d.id: d for d in corpus.split_documents(lines, "gdpr", "http://x")}
    assert list(docs) == ["gdpr-art-1", "gdpr-art-2", "gdpr-art-3", "gdpr-art-3a", "gdpr-annex-i"]
    assert docs["gdpr-art-1"].title == "Subject matter"
    assert docs["gdpr-art-1"].text.startswith("1. This Regulation lays down rules")
    assert "(b) ‘owner’ means a person" in docs["gdpr-art-2"].text
    assert docs["gdpr-art-3a"].number == "3a" and docs["gdpr-art-3a"].heading == "GDPR Article 3a - Lettered article"
    assert docs["gdpr-annex-i"].kind == "annex" and docs["gdpr-annex-i"].title == "List of widget types"
    assert docs["gdpr-art-1"].source_url == "http://x"


def test_structure_headings_and_signature_do_not_leak_into_articles(lines):
    docs = {d.id: d for d in corpus.split_documents(lines, "gdpr")}
    assert "CHAPTER" not in docs["gdpr-art-2"].text and "Obligations" not in docs["gdpr-art-2"].text
    assert "Done at" not in docs["gdpr-art-3a"].text and "Brussels" not in docs["gdpr-art-3a"].text


def test_inline_mentions_do_not_start_articles(lines):
    assert not any(d.number == "5" for d in corpus.split_documents(lines, "gdpr"))


def test_duplicate_headings_keep_the_longest_text():
    lines = ["Article 1", "Short title", "tiny", "Article 1", "Short title", "a much longer body text here"]
    (doc,) = corpus.split_documents(lines, "gdpr")
    assert doc.text == "a much longer body text here"


def test_title_is_not_taken_from_a_paragraph():
    docs = corpus.split_documents(["Article 7", "1. The first paragraph starts here.", "More text."], "gdpr")
    assert docs[0].title == "" and docs[0].text.startswith("1. The first paragraph")


def test_missing_articles_reports_gaps(lines, monkeypatch):
    monkeypatch.setitem(corpus.LAWS["gdpr"], "expected_articles", 4)
    docs = corpus.split_documents(lines, "gdpr")
    assert corpus.missing_articles(docs, "gdpr") == [4]


def test_parse_law_validates_article_count(monkeypatch):
    html = FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setitem(corpus.LAWS["gdpr"], "expected_articles", 3)
    assert len(corpus.parse_law(html, "gdpr")) == 5
    monkeypatch.setitem(corpus.LAWS["gdpr"], "expected_articles", 99)
    with pytest.raises(ValueError, match="missing 96"):
        corpus.parse_law(html, "gdpr")


def test_build_corpus_from_saved_html(tmp_path, monkeypatch):
    monkeypatch.setitem(corpus.LAWS["gdpr"], "expected_articles", 3)
    docs = corpus.build_corpus("gdpr", tmp_path, html_path=FIXTURE)
    assert (tmp_path / "raw" / "gdpr.html").exists()
    assert [d.id for d in load_docs(tmp_path / "processed" / "gdpr.jsonl")] == [d.id for d in docs]
    corpus.write_sources(tmp_path, ["gdpr"], {"gdpr": len(docs)})
    text = (tmp_path / "SOURCES.md").read_text()
    assert "32016R0679" in text and "2011/833/EU" in text


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body.encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_html_success_and_blocked(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(200, "<p>Article 1</p>"))
    body, url = corpus.fetch_html("gdpr")
    assert "Article 1" in body and "32016R0679" in url
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(202, ""))
    with pytest.raises(corpus.FetchBlockedError, match="--html"):
        corpus.fetch_html("gdpr")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(200, "<p>captcha</p>"))
    with pytest.raises(corpus.FetchBlockedError):
        corpus.fetch_html("aiact")


def test_fetch_html_falls_back_to_cellar_after_waf(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=60.0):
        calls.append(req.full_url)
        if "eur-lex.europa.eu" in req.full_url:
            return _Resp(202, "")
        return _Resp(200, "<p>Article 1 from Cellar</p>")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    body, url = corpus.fetch_html("gdpr")
    assert "Article 1 from Cellar" in body
    assert "publications.europa.eu/resource/celex/32016R0679.ENG" in url
    assert any("eur-lex.europa.eu" in u for u in calls)


def test_fetch_html_network_error(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(corpus.FetchBlockedError, match="offline"):
        corpus.fetch_html("gdpr")
