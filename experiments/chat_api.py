#%%
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os
from datetime import datetime
from cerebras.cloud.sdk import Cerebras
from twilio.rest import Client as TwilioClient

import json
import time


# LLM setup
load_dotenv()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
SYSTEM_PROMPT = (Path(__file__).parent / 'system_prompt.txt').read_text()
RESUME = (Path(__file__).parent / 'resume.txt').read_text()
# MODEL = "gpt-5-nano"
# MODEL = "qwen-3-32b"
# MODEL = "llama-3.3-70b"
MODEL = "gpt-oss-120b"

model_client = Cerebras(api_key=CEREBRAS_API_KEY)
twilio_client = TwilioClient(username=TWILIO_ACCOUNT_SID, password=TWILIO_AUTH_TOKEN)

# HTTP requests
class ChatMessage(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
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
        {"role": "system", "content": RESUME}
    ]
    base.extend(m.model_dump() for m in user_messages)
    return base


# logging setup
#  1. modify chat endpoint to:
#   - record start timestamp before streaming begins
#   - track bot response by accumulating chunks
#   - record end timestamp after stream completes
#  2. write log entry as JSONl with fields:
#   - timestamp: ISO format string
#   - user_message: string content
#   - bot_response: full accumulated response string
#   - latency_ms: float(end-start) * 1000


LOG_PATH = Path(os.getenv("LOG_PATH", "experiments/logs/chat_logs.jsonl"))
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
current_time = datetime.utcnow().isoformat()

def write_log_entry(user_message: str, bot_response: str, token_latency_ms: float, message_count: int, token_count: int) -> None:
    """write a single log entry to a JSONL file"""
    entry = {
        "user_message": user_message, 
        "bot_response": bot_response,
        "token_latency_ms": token_latency_ms,
        "timestamp": datetime.utcnow().isoformat(),
        "message_count": message_count,
        "token_count": token_count
    }
    try:
        with LOG_PATH.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        print(f"logging failed with error:\n\n {e}")

# write_log_entry('test message', 'the message worked', -1., 0)
# write_log_entry('test message 2', 'its a success', -1, 1)

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
            if hasattr(chunk, 'usage') and chunk.usage and chunk.usage.completion_tokens:
                token_count = chunk.usage.completion_tokens
        if token_count == -1:
            token_count = chunk_count
        latency = (time.time() - start)*1000
        
        message_count = len(request.messages)
        user_message = request.messages[-1].content if request.messages else ''
        token_latency_ms = latency / token_count if token_count not in [0, -1] else -1
        write_log_entry(
            user_message=user_message, 
            bot_response=bot_response, 
            token_latency_ms=token_latency_ms,
            message_count=message_count,
            token_count = token_count
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
