"""Paragraph-aware chunking. Sizes are in words (about 0.75 words per token, so 300 words is roughly 400 tokens)."""

from __future__ import annotations

from eurag.schema import Chunk, Doc


def _windows(words: list[str], size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def chunk_doc(doc: Doc, max_words: int = 300, overlap_words: int = 40) -> list[Chunk]:
    """Pack whole paragraphs up to ``max_words``; split a single oversize paragraph with word overlap.

    Each chunk starts with the document heading (for example ``GDPR Article 17 - Right to erasure``) so
    that the passage carries its own provenance into the embedding and into the prompt.
    """
    if max_words < 20:
        raise ValueError("max_words must be at least 20")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")
    pieces: list[str] = []
    for para in (p for p in doc.text.split("\n") if p.strip()):
        words = para.split()
        if len(words) > max_words:
            pieces.extend(_windows(words, max_words, overlap_words))
        else:
            pieces.append(para.strip())
    bodies: list[str] = []
    current: list[str] = []
    count = 0
    for piece in pieces:
        n = len(piece.split())
        if current and count + n > max_words:
            bodies.append("\n".join(current))
            current, count = [], 0
        current.append(piece)
        count += n
    if current:
        bodies.append("\n".join(current))
    return [
        Chunk(f"{doc.id}#{i}", doc.id, doc.law, doc.heading, f"{doc.heading}\n{body}", i)
        for i, body in enumerate(bodies)
    ]


def chunk_corpus(docs: list[Doc], max_words: int = 300, overlap_words: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_doc(doc, max_words, overlap_words))
    return chunks
