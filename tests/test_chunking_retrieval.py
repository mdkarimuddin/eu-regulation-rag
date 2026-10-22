import numpy as np
import pytest
from conftest import doc_id, mini_docs

from eurag.bm25 import BM25, tokenize
from eurag.chunking import chunk_corpus, chunk_doc
from eurag.embed import HashingEmbedder
from eurag.retrieve import RetrievalConfig, doc_ranking, rrf, search_chunks
from eurag.schema import Doc


def long_doc(paragraphs, words_each):
    text = "\n".join(" ".join(f"w{p}x{i}" for i in range(words_each)) for p in range(paragraphs))
    return Doc("gdpr-art-9", "gdpr", "article", "9", "Long article", text)


def test_chunk_has_heading_and_stable_ids():
    (chunk,) = chunk_doc(mini_docs()[0])
    assert chunk.text.startswith("Test Article 1 - Duty to register\n")
    assert chunk.id == "test-art-1#0" and chunk.doc_id == "test-art-1" and chunk.index == 0


def test_paragraphs_are_packed_up_to_the_limit():
    chunks = chunk_doc(long_doc(paragraphs=6, words_each=100), max_words=250, overlap_words=20)
    assert len(chunks) == 3  # 2 paragraphs (200 words) per chunk
    assert all(len(c.text.split("\n", 1)[1].split()) <= 250 for c in chunks)


def test_oversize_paragraph_is_split_with_overlap():
    chunks = chunk_doc(long_doc(paragraphs=1, words_each=500), max_words=200, overlap_words=50)
    bodies = [c.text.split("\n", 1)[1].split() for c in chunks]
    assert len(bodies) >= 3 and all(len(b) <= 200 for b in bodies)
    assert bodies[0][-50:] == bodies[1][:50]  # consecutive windows share the overlap
    assert set(sum(bodies, [])) == {f"w0x{i}" for i in range(500)}  # nothing lost


def test_chunk_parameter_validation_and_empty_doc():
    with pytest.raises(ValueError):
        chunk_doc(mini_docs()[0], max_words=5)
    with pytest.raises(ValueError):
        chunk_doc(mini_docs()[0], max_words=100, overlap_words=100)
    assert chunk_doc(Doc("x", "gdpr", "article", "1", "t", "   \n  ")) == []


def test_chunk_corpus_covers_every_doc():
    docs = mini_docs()
    assert {c.doc_id for c in chunk_corpus(docs)} == {d.id for d in docs}


def test_tokenize_and_bm25_ranking():
    assert tokenize("Widgets, 30 DAYS!") == ["widgets", "30", "days"]
    bm = BM25(["the cat sat", "the dog barked loudly", "cats and dogs"])
    hits = bm.search("dog barked", 3)
    assert hits[0][0] == 1 and all(score > 0 for _, score in hits)
    assert bm.search("zebra", 3) == []
    assert BM25([]).search("x", 3) == []


def test_bm25_idf_prefers_rare_terms():
    bm = BM25(["common rare", "common common", "common other"])
    assert bm.search("rare common", 3)[0][0] == 0


def test_rrf_fusion_properties():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert [i for i, _ in fused][:2] == [1, 3]  # items ranked well by both lists win
    assert {i for i, _ in fused} == {1, 2, 3, 4}
    assert rrf([[5]]) == [(5, 1 / 61)]
    assert rrf([]) == []


def test_doc_ranking_collapses_chunks_in_first_seen_order(index):
    ids = doc_ranking(index.chunks, list(range(len(index.chunks))))
    assert ids == [c.doc_id for c in index.chunks]  # one chunk per mini doc
    dup = [0, 0, 1]
    assert doc_ranking(index.chunks, dup) == [index.chunks[0].doc_id, index.chunks[1].doc_id]


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_each_mode_finds_the_right_article(index, embedder, mode):
    cfg = RetrievalConfig(mode=mode)
    hits = search_chunks(index, embedder, "within how many days must an owner register a widget", cfg)
    assert index.chunks[hits[0][0]].doc_id == doc_id("registration")


def test_candidates_limit_and_bm25_no_match(index, embedder):
    assert len(search_chunks(index, embedder, "widget", RetrievalConfig(mode="hybrid", candidates=3))) == 3
    assert search_chunks(index, embedder, "zzzzqqq", RetrievalConfig(mode="bm25")) == []


def test_hybrid_rescues_a_query_the_hashing_vector_misses(index, embedder):
    # a rare exact term decides BM25; fusion keeps the BM25 winner near the top
    hits = search_chunks(index, embedder, "fireproof", RetrievalConfig(mode="hybrid"))
    assert index.chunks[hits[0][0]].doc_id == doc_id("transport")


def test_config_validation_and_label():
    with pytest.raises(ValueError):
        RetrievalConfig(mode="magic")
    with pytest.raises(ValueError):
        RetrievalConfig(candidates=0)
    assert RetrievalConfig(mode="hybrid", rerank=True, rewrite=True).label() == "hybrid+rerank+rewrite"


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder()
    a = emb.embed(["widget registration", "widget registration"])
    assert np.allclose(a[0], a[1]) and np.isclose(np.linalg.norm(a[0]), 1.0)
    assert emb.embed([""]).sum() == 0
    near, far = emb.embed(["register widget now", "widget register", "completely different words"])[[0, 1]], None
    sim_near = float(near[0] @ near[1])
    far = emb.embed(["completely different words"])[0]
    assert sim_near > float(near[0] @ far)
