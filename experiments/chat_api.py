#%%
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os
from openai import OpenAI

# LLM setup
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT = (Path(__file__).parent / 'system_prompt.txt').read_text()
RESUME = (Path(__file__).parent / 'resume.txt').read_text()
MODEL = "gpt-5-nano"
client = OpenAI()

def call_llm(user_message: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Here is Alex's resume:\n\n {RESUME}"},
            {"role": "user", "content": user_message},
        ]
    )
    return completion.choices[0].message.content


# HTTP requests
class ChatRequest(BaseModel):
    message: str
class ChatResponse(BaseModel):
    reply: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4000",
        "http://127.0.0.1:4000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    reply_text = call_llm(request.message)
    return ChatResponse(reply=reply_text)
    

@app.post("/chat-stream")
def chat_stream(request: ChatRequest) -> ChatResponse:
    def token_stream():
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": RESUME},
                {"role": "user", "content": request.message}

            ],
            stream=True
        )
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    return StreamingResponse(token_stream(), media_type="text/plain")
