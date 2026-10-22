"""Integration test for the pgvector backend. Runs only when EURAG_PG_DSN points at Postgres with pgvector
(the CI workflow provides one; locally: docker compose up -d db)."""

import os

import pytest
from conftest import doc_id, mini_docs

from eurag.chunking import chunk_corpus
from eurag.embed import HashingEmbedder
from eurag.pipeline import RagPipeline
from eurag.retrieve import RetrievalConfig
from eurag.store import MemoryIndex, PgIndex

pytestmark = pytest.mark.pg
DSN = os.environ.get("EURAG_PG_DSN")


@pytest.mark.skipif(not DSN, reason="EURAG_PG_DSN not set")
def test_pgvector_matches_memory_index():
    emb = HashingEmbedder()
    mem = MemoryIndex.build(chunk_corpus(mini_docs()), emb)
    pg = PgIndex(DSN, emb.dim)
    pg.replace_all(mem.chunks, mem.vectors)
    assert [c.id for c in pg.chunks] == [c.id for c in mem.chunks]
    q = emb.embed(["fireproof packaging for batteries"], is_query=True)[0]
    assert pg.vector_search(q, 3)[0][0] == mem.vector_search(q, 3)[0][0]
    pipe = RagPipeline(pg, emb, config=RetrievalConfig(mode="hybrid"))
    assert pipe.retrieve("fireproof packaging for batteries").doc_ids[0] == doc_id("transport")
