#%%
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os
from cerebras.cloud.sdk import Cerebras

# LLM setup
load_dotenv()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
SYSTEM_PROMPT = (Path(__file__).parent / 'system_prompt.txt').read_text()
RESUME = (Path(__file__).parent / 'resume.txt').read_text()
# MODEL = "gpt-5-nano"
# MODEL = "qwen-3-32b"
# MODEL = "llama-3.3-70b"
MODEL = "gpt-oss-120b"

client = Cerebras(api_key=CEREBRAS_API_KEY)


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


@app.post("/chat")
def chat_stream(request: ChatRequest) -> ChatResponse:
    def token_stream():
        completion = client.chat.completions.create(
            model=MODEL,
            messages=build_messages(request.messages),
            stream=True
        )
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    return StreamingResponse(token_stream(), media_type="text/plain")
