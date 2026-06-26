FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv --version

ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --only-group prod

COPY experiments/ ./experiments/

# Pre-download ChromaDB's ONNX embedding model so it's cached in the image
# instead of being downloaded on every cold start (~80MB, adds ~14s to startup)
RUN uv run --no-sync python -c "import chromadb; c = chromadb.Client(); col = c.get_or_create_collection('warmup'); col.add(documents=['warmup'], ids=['1']); col.query(query_texts=['warmup'], n_results=1)"

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "uvicorn", "experiments.chat_api:app", "--host", "0.0.0.0", "--port", "8080"]
