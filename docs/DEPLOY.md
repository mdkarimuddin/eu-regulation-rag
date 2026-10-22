# Deployment (not done yet)

Pick one small host (Render, Fly.io or a VPS). The image builds the index at build time, so the container needs no
database to start.

1. Commit `data/processed/*.jsonl` (the processed text) so the Docker build can index it.
2. Build and run locally first: `docker build -t eu-regulation-rag . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY eu-regulation-rag`
   (passing `-e ANTHROPIC_API_KEY` with no value forwards it from your shell).
3. On the host, set `ANTHROPIC_API_KEY` as a secret in the host's settings. Do not put it in the repo or the image.
4. Leave `EURAG_INGEST_TOKEN` unset unless you need `/ingest`; the endpoint is disabled without it.
5. Add a spending limit to the API key in the Anthropic console before making the URL public, and consider a
   per-IP rate limit at the host or a reverse proxy, because `/ask` has no authentication.
6. Put the live URL in the README only after `/health` and one real `/ask` have worked from outside your network.

To use Postgres instead of files: `docker compose up -d db`, then
`EURAG_PG_DSN=postgresql://postgres:eurag@localhost:5432/eurag eurag build-index --embedder st --pg-dsn $EURAG_PG_DSN`.
