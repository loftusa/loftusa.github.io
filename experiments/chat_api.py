from pathlib import Path
import os
import json
import time
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel
from cerebras.cloud.sdk import Cerebras
from fastapi import Header, HTTPException, FastAPI
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware


# LLM setup + constants
# Load .env from experiments directory (or parent directory)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()  # Also try parent directory .env for backward compatibility
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()
RESUME = (Path(__file__).parent / "resume.txt").read_text()
LOG_PATH = Path(os.getenv("LOG_PATH", "/app/experiments/logs/chat_logs.jsonl"))
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
LOG_ACCESS_TOKEN = os.getenv("LOG_ACCESS_TOKEN")
MODEL = "gpt-oss-120b"

model_client = Cerebras(api_key=CEREBRAS_API_KEY)


# HTTP requests pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str | None = None
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    reply: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4000",
        "http://127.0.0.1:4000",
        "https://alex-loftus.com",
        "https://www.alex-loftus.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_messages(user_messages: list[ChatMessage]) -> list[dict[str, str]]:
    base: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": RESUME},
    ]
    base.extend(m.model_dump() for m in user_messages)
    return base


def write_log_entry(
    user_message: str,
    bot_response: str,
    token_latency_ms: float,
    message_count: int,
    token_count: int,
    user_id: str | None,
) -> None:
    """write a single log entry to a JSONL file"""
    entry = {
        "user_id": user_id,
        "user_message": user_message,
        "bot_response": bot_response,
        "token_latency_ms": token_latency_ms,
        "timestamp": datetime.utcnow().isoformat(),
        "message_count": message_count,
        "token_count": token_count,
    }
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"logging failed with error:\n\n {e}")


@app.post("/chat")
def chat_stream(request: ChatRequest) -> ChatResponse:
    """
    todo:
      - record start timestamp before streaming begins
      - track bot response by accumulating chunks
      - record end timestamp after stream completes
    """

    def token_stream():
        completion = model_client.chat.completions.create(
            model=MODEL, 
            messages=build_messages(request.messages), 
            stream=True
        )
        bot_response = ""
        token_count = -1
        start = time.time()
        chunk_count = 0
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                bot_response += delta.content
                chunk_count += 1
                yield delta.content
            if (
                hasattr(chunk, "usage")
                and chunk.usage
                and chunk.usage.completion_tokens
            ):
                token_count = chunk.usage.completion_tokens
        if token_count == -1:
            token_count = chunk_count
        latency = (time.time() - start) * 1000

        message_count = len(request.messages)
        user_message = request.messages[-1].content if request.messages else ""
        token_latency_ms = latency / token_count if token_count not in [0, -1] else -1
        write_log_entry(
            user_message=user_message,
            bot_response=bot_response,
            token_latency_ms=token_latency_ms,
            message_count=message_count,
            token_count=token_count,
            user_id=request.user_id,
        )

    return StreamingResponse(token_stream(), media_type="text/plain")


# health check
@app.get("/health")
def health_check() -> dict[str, bool | str]:
    return {
        "status": "ok",
        "api_key": bool(CEREBRAS_API_KEY),
        "timestamp": datetime.utcnow().isoformat(),
    }


# log access
@app.get("/logs/download")
def download_logs(authorization: str | None = Header(None)):
    """
    Authorization: Bearer <token>
    """
    if LOG_ACCESS_TOKEN is None:
        raise HTTPException(status_code=500)
    if not authorization:
        raise HTTPException(status_code=401)
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401)
    auth_token = authorization.removeprefix("Bearer ").strip()
    if auth_token != LOG_ACCESS_TOKEN:
        raise HTTPException(status_code=401)
    if not LOG_PATH.exists():
        raise HTTPException(status_code=404)
    if LOG_PATH.stat().st_size == 0:
        return Response(content="", media_type="text/plain")

    return FileResponse(LOG_PATH)
