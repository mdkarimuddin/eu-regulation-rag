"""Fetch EUR-Lex HTML and split it into one document per article/annex.

The splitter works on plain text (not on EUR-Lex's CSS classes), so it keeps working if the markup changes.
Every parse is checked against the expected article numbers and fails loudly when articles are missing.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

from eurag.schema import Doc, make_doc_id, save_docs

LAWS = {
    "gdpr": {
        "celex": "32016R0679",
        "title": "Regulation (EU) 2016/679 (General Data Protection Regulation)",
        "expected_articles": 99,
    },
    "aiact": {
        "celex": "32024R1689",
        "title": "Regulation (EU) 2024/1689 (Artificial Intelligence Act), original Official Journal text",
        "expected_articles": 113,
    },
}
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
# Same Official Journal text, via the Publications Office Cellar. EUR-Lex HTML is often
# behind an AWS WAF challenge (HTTP 202) for non-browser clients.
CELLAR_URL = "https://publications.europa.eu/resource/celex/{celex}.ENG"
LICENCE_NOTE = (
    "EU legislation published on EUR-Lex may be reused with acknowledgement of the source "
    "(Commission Decision 2011/833/EU). Check the current EUR-Lex legal notice before redistributing."
)

_BLOCK = {"p", "div", "tr", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "ul", "ol"}
_SKIP = {"script", "style", "head", "noscript"}
_MARKER = re.compile(r"^(\(\w{1,4}\)|\d{1,3}\.|[–—-])$")
_ARTICLE = re.compile(r"^Article\s+(\d+)([a-z]?)$")
_ANNEX = re.compile(r"^ANNEX\s+([IVXLCDM]+)$")
_STRUCTURE = re.compile(r"^(CHAPTER|SECTION|TITLE)\s+[IVXLCDM\d]+$")


class FetchBlockedError(RuntimeError):
    """EUR-Lex did not return the law text (bot protection or network problem)."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_lines(html: str) -> list[str]:
    parser = _TextExtractor()
    parser.feed(html)
    raw = "".join(parser.parts).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.split("\n")]
    lines = [ln for ln in lines if ln]
    merged: list[str] = []
    i = 0
    while i < len(lines):  # "(a)" on its own line belongs with the text that follows it
        if _MARKER.match(lines[i]) and i + 1 < len(lines) and not _MARKER.match(lines[i + 1]):
            merged.append(f"{lines[i]} {lines[i + 1]}")
            i += 2
        else:
            merged.append(lines[i])
            i += 1
    return merged


def split_documents(lines: list[str], law: str, source_url: str = "") -> list[Doc]:
    """Split cleaned lines into article and annex documents. Duplicates keep the longest version."""
    found: dict[str, Doc] = {}
    current: tuple[str, str, str] | None = None  # (kind, number, title)
    body: list[str] = []

    def flush() -> None:
        nonlocal current, body
        if current is not None and body:
            kind, number, title = current
            doc_id = make_doc_id(law, kind, number)
            doc = Doc(doc_id, law, kind, number, title, "\n".join(body).strip(), source_url)
            if doc_id not in found or len(doc.text) > len(found[doc_id].text):
                found[doc_id] = doc
        current, body = None, []

    i = 0
    while i < len(lines):
        line = lines[i]
        art, annex = _ARTICLE.match(line), _ANNEX.match(line)
        if art or annex:
            flush()
            kind = "article" if art else "annex"
            number = (art.group(1) + art.group(2)) if art else annex.group(1)
            title = ""
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if len(nxt) <= 200 and not nxt.endswith((".", ";", ":")) and not re.match(r"^(\(\w+\)|\d+\.)", nxt):
                    title = nxt.rstrip("`").strip()
                    i += 1
            current = (kind, number, title)
        elif _STRUCTURE.match(line):
            flush()
            i += 1  # the line after "CHAPTER III" is the chapter title
        elif line.startswith("Done at"):
            flush()
        elif current is not None:
            body.append(line)
        i += 1
    flush()
    return sorted(found.values(), key=lambda d: (d.kind != "article", _sort_key(d.number)))


def _sort_key(number: str) -> tuple[int, str]:
    m = re.match(r"^(\d+)(.*)$", number)
    return (int(m.group(1)), m.group(2)) if m else (10_000, number)


def missing_articles(docs: list[Doc], law: str) -> list[int]:
    have = {int(re.match(r"\d+", d.number).group()) for d in docs if d.kind == "article"}
    return [n for n in range(1, LAWS[law]["expected_articles"] + 1) if n not in have]


def parse_law(html: str, law: str, source_url: str = "") -> list[Doc]:
    """Parse and validate. Raises ValueError listing missing article numbers."""
    docs = split_documents(html_to_lines(html), law, source_url)
    missing = missing_articles(docs, law)
    if missing:
        shown = ", ".join(str(n) for n in missing[:20])
        raise ValueError(
            f"{law}: parsed {sum(d.kind == 'article' for d in docs)} articles, "
            f"missing {len(missing)} (e.g. {shown}). The page may be incomplete or its layout changed."
        )
    return docs


def _looks_like_law(html: str) -> bool:
    return "Article 1" in html or "Article\xa01" in html or "Article&nbsp;1" in html


def _download(url: str, headers: dict[str, str], timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed EU Publications URLs
        return resp.status, resp.read().decode("utf-8", errors="replace")


def fetch_html(law: str, timeout: float = 60.0) -> tuple[str, str]:
    """Download Official Journal HTML. Tries EUR-Lex, then the Cellar CELEX expression."""
    celex = LAWS[law]["celex"]
    eurlex = EURLEX_URL.format(celex=celex)
    cellar = CELLAR_URL.format(celex=celex)
    ua = {"User-Agent": "Mozilla/5.0 (research; eu-regulation-rag)", "Accept-Language": "en"}
    errors: list[str] = []
    try:
        status, body = _download(eurlex, ua, timeout)
        if status == 200 and _looks_like_law(body):
            return body, eurlex
        errors.append(f"EUR-Lex HTTP {status}, law text missing")
    except (urllib.error.URLError, TimeoutError) as exc:
        errors.append(f"EUR-Lex: {exc}")
    try:
        status, body = _download(
            cellar, {**ua, "Accept": "application/xhtml+xml,text/html"}, timeout
        )
        if status == 200 and _looks_like_law(body):
            return body, cellar
        errors.append(f"Cellar HTTP {status}, law text missing")
    except (urllib.error.URLError, TimeoutError) as exc:
        errors.append(f"Cellar: {exc}")
    raise FetchBlockedError(
        f"could not download {law} ({'; '.join(errors)}). "
        f"Open {eurlex} in a browser, save the page as HTML, then run: eurag fetch --law {law} --html FILE"
    )


def build_corpus(law: str, out_dir: str | Path, html_path: str | Path | None = None) -> list[Doc]:
    out = Path(out_dir)
    if html_path:
        html = Path(html_path).read_text(encoding="utf-8", errors="replace")
        url = EURLEX_URL.format(celex=LAWS[law]["celex"])
    else:
        html, url = fetch_html(law)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "raw" / f"{law}.html").write_text(html, encoding="utf-8")
    docs = parse_law(html, law, url)
    save_docs(out / "processed" / f"{law}.jsonl", docs)
    return docs


def write_sources(out_dir: str | Path, laws: list[str], counts: dict[str, int]) -> None:
    lines = ["# Sources and licence", "", f"Retrieved: {date.today().isoformat()}", ""]
    for law in laws:
        meta = LAWS[law]
        lines.append(f"- {meta['title']}: {EURLEX_URL.format(celex=meta['celex'])} ({counts.get(law, 0)} documents)")
    lines += ["", LICENCE_NOTE, ""]
    Path(out_dir, "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")
