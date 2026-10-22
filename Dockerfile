FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
# git is required because llm-eval-harness is installed from GitHub, not PyPI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir ".[embeddings,anthropic]" \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
# Build the index at image build time so the container starts fast. Needs data/processed/*.jsonl in the build context.
RUN eurag build-index --processed data/processed --embedder st --out /app/index
ENV EURAG_INDEX=/app/index EURAG_EMBEDDER=st
EXPOSE 8000
CMD ["eurag", "serve", "--host", "0.0.0.0", "--port", "8000"]
