FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv --version

ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml .
RUN uv sync

COPY experiments/ ./experiments/

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "experiments.chat_api:app", "--host", "0.0.0.0", "--port", "8080"]
