"""FastAPI service: /ask, /health, /ingest (token-protected) and a minimal HTML page."""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from eurag.chunking import chunk_corpus
from eurag.pipeline import RagPipeline
from eurag.schema import Doc
from eurag.store import MemoryIndex

PAGE = """<!doctype html><meta charset="utf-8"><title>EU AI Act and GDPR - cited answers</title>
<style>body{font-family:system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:5rem}pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem}</style>
<h1>Ask about the EU AI Act and the GDPR</h1>
<p>Answers cite article ids. This describes the text of the law and is not legal advice.</p>
<textarea id="q" maxlength="500" placeholder="For example: Within how many hours must a personal data breach be reported?"></textarea>
<p><button onclick="go()">Ask</button></p><pre id="out"></pre>
<script>
async function go(){const out=document.getElementById('out');out.textContent='...';
const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({question:document.getElementById('q').value})});
const j=await r.json();out.textContent=r.ok?j.answer+'\\n\\nCited: '+j.cited.join(', ')+'\\nTop sources: '+j.sources.slice(0,5).join(', '):JSON.stringify(j)}
</script>"""


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    cited: list[str]
    sources: list[str]
    latency_ms: float
    tokens_in: int
    tokens_out: int


class DocIn(BaseModel):
    id: str
    law: str
    kind: str = "article"
    number: str
    title: str = ""
    text: str = Field(min_length=1)
    source_url: str = ""


class IngestRequest(BaseModel):
    docs: list[DocIn] = Field(min_length=1, max_length=200)


def add_docs(pipeline: RagPipeline, docs: list[Doc], chunk_words: int = 300, overlap: int = 40) -> int:
    """Append documents to a memory index (replacing any with the same id) and rebuild BM25."""
    old = pipeline.index
    new_ids = {d.id for d in docs}
    keep = [i for i, c in enumerate(old.chunks) if c.doc_id not in new_ids]
    new_chunks = chunk_corpus(docs, chunk_words, overlap)
    new_vecs = pipeline.embedder.embed([c.text for c in new_chunks])
    chunks = [old.chunks[i] for i in keep] + new_chunks
    vectors = np.vstack([old.vectors[keep], new_vecs]) if keep else new_vecs
    pipeline.index = MemoryIndex(chunks, vectors, {**old.meta, "n_chunks": len(chunks)})
    return len(new_chunks)


def create_app(get_pipeline: Callable[[], RagPipeline]) -> FastAPI:
    app = FastAPI(title="eu-regulation-rag", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return PAGE

    @app.get("/health")
    def health() -> dict:
        try:
            p = get_pipeline()
        except Exception as exc:  # noqa: BLE001 - report any startup problem as unhealthy
            raise HTTPException(503, f"pipeline unavailable: {type(exc).__name__}") from exc
        return {"status": "ok", "chunks": len(p.index.chunks), "embedder": p.index.meta.get("embedder")}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        try:
            pipeline = get_pipeline()
            a = pipeline.ask(req.question)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak internals to the caller
            raise HTTPException(503, f"could not answer: {type(exc).__name__}") from exc
        return AskResponse(
            answer=a.answer,
            abstained=a.abstained,
            cited=a.cited,
            sources=a.sources[:10],
            latency_ms=a.latency_ms,
            tokens_in=a.tokens_in,
            tokens_out=a.tokens_out,
        )

    @app.post("/ingest")
    def ingest(req: IngestRequest, x_ingest_token: str | None = Header(default=None)) -> dict:
        expected = os.environ.get("EURAG_INGEST_TOKEN", "")
        if not expected:
            raise HTTPException(403, "ingest is disabled: set EURAG_INGEST_TOKEN to enable it")
        if not x_ingest_token or not hmac.compare_digest(x_ingest_token, expected):
            raise HTTPException(401, "invalid ingest token")
        pipeline = get_pipeline()
        docs = [Doc(**d.model_dump()) for d in req.docs]
        added = add_docs(pipeline, docs)
        return {"documents": len(docs), "chunks_added": added, "chunks_total": len(pipeline.index.chunks)}

    return app


def app_from_env() -> FastAPI:
    holder: dict[str, RagPipeline] = {}

    def getter() -> RagPipeline:
        if "p" not in holder:
            from eurag.config import pipeline_from_env

            holder["p"] = pipeline_from_env()
        return holder["p"]

    return create_app(getter)
