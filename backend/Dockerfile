# Backend image for alex-loftus.com — built from the REPO ROOT (fly.toml sets dockerfile = backend/Dockerfile).
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install prod deps from the backend manifest (no torch). Done before copying code for layer cache.
COPY backend/pyproject.toml ./backend/pyproject.toml
RUN uv pip install --system -r backend/pyproject.toml

# App code: the backend package + the experiments library it imports (pure fold logic, RAG module,
# system_prompt/resume, seeds.json, and the prebaked Chroma collection under experiments/rag/data).
COPY backend/ ./backend/
COPY experiments/ ./experiments/

RUN chmod +x /app/backend/entrypoint.sh

# Pre-download ChromaDB's ONNX embedding model so it's cached in the image (avoids ~14s cold start).
RUN python -c "import chromadb; c = chromadb.Client(); col = c.get_or_create_collection('warmup'); col.add(documents=['warmup'], ids=['1']); col.query(query_texts=['warmup'], n_results=1)"

EXPOSE 8080

# entrypoint runs `alembic upgrade head` (SQLite schema on the volume) then launches uvicorn.
ENTRYPOINT ["/app/backend/entrypoint.sh"]
