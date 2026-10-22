FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir ".[embeddings,anthropic]"
# Build the index at image build time so the container starts fast. Needs data/processed/*.jsonl in the build context.
RUN eurag build-index --processed data/processed --embedder st --out /app/index
ENV EURAG_INDEX=/app/index EURAG_EMBEDDER=st
EXPOSE 8000
CMD ["eurag", "serve", "--host", "0.0.0.0", "--port", "8000"]
