"""Shared fixtures. The mini corpus is invented text about a fictional 'Widget Act'; it is NOT real law."""

from __future__ import annotations

import pytest

from eurag.chunking import chunk_corpus
from eurag.embed import HashingEmbedder
from eurag.llm import LLMReply
from eurag.pipeline import RagPipeline
from eurag.retrieve import RetrievalConfig
from eurag.schema import Doc
from eurag.store import MemoryIndex

TOPICS = {
    "registration": ("Duty to register", "Every owner must register each widget with the widget registry within 30 days of purchase."),
    "recall": ("Safety recalls", "A manufacturer must recall any widget that overheats and must notify owners of the recall within 48 hours."),
    "fines": ("Penalties", "Selling an unregistered widget is punished by a fine of up to 5000 euros per widget."),
    "warranty": ("Warranty", "Every widget carries a warranty of three years covering manufacturing defects but not accidental damage."),
    "import": ("Imports", "Importers must verify that imported widgets carry the conformity label before placing them on the market."),
    "labelling": ("Labelling", "Each widget shall bear a label showing the serial number, the manufacturer and the voltage rating."),
    "privacy": ("Owner privacy", "The registry shall not publish the names or addresses of widget owners."),
    "inspection": ("Inspections", "Inspectors may enter a warehouse during business hours to check widgets and records."),
    "appeal": ("Appeals", "A manufacturer may appeal a recall order to the widget tribunal within 14 days."),
    "transport": ("Transport", "Widgets containing batteries must be transported in fireproof packaging."),
}


def mini_docs() -> list[Doc]:
    return [
        Doc(f"test-art-{i}", "test", "article", str(i), title, text)
        for i, (title, text) in enumerate(TOPICS.values(), start=1)
    ]


def doc_id(topic: str) -> str:
    return f"test-art-{list(TOPICS).index(topic) + 1}"


class ScriptedLLM:
    """Returns text from a function(system, user, max_tokens); records every call."""

    name = "scripted"

    def __init__(self, fn=None):
        self.calls: list[tuple[str, str]] = []
        self._fn = fn or (lambda system, user, max_tokens: "NOT_FOUND")

    def complete(self, system, user, max_tokens=600):
        self.calls.append((system, user))
        return LLMReply(self._fn(system, user, max_tokens), tokens_in=100, tokens_out=20)


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def index(embedder):
    return MemoryIndex.build(chunk_corpus(mini_docs(), 300, 40), embedder)


@pytest.fixture
def make_pipeline(index, embedder):
    def make(mode="hybrid", llm=None, reranker=None, **cfg):
        return RagPipeline(index, embedder, llm=llm, reranker=reranker, config=RetrievalConfig(mode=mode, **cfg))

    return make
