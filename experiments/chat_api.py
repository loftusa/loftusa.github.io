from dataclasses import dataclass, field
from pathlib import Path
import os
import json
import time
import uuid
from collections import OrderedDict
from datetime import datetime

from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from cerebras.cloud.sdk import Cerebras
from fastapi import Header, HTTPException, FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from experiments.rag_context import load_collection, retrieve_context, format_rag_context
from experiments.coauthorship.overrides import fold_events
from experiments.coauthorship.affiliation_events import (
    ENTRY_TYPES, FIELD_CAPS as AFF_FIELD_CAPS, fold_aff_events, norm_person)


# LLM setup + constants
# Load .env from experiments directory (or parent directory)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()  # Also try parent directory .env for backward compatibility
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL = "zai-glm-4.7"
SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()
RESUME = (Path(__file__).parent / "resume.txt").read_text()
SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + RESUME}
LOG_PATH = Path(os.getenv("LOG_PATH", "/app/experiments/logs/chat_logs.jsonl"))
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
LOG_ACCESS_TOKEN = os.getenv("LOG_ACCESS_TOKEN")

# co-authorship graph corrections (see experiments/coauthorship/overrides.py + merge_corrections.py).
# Append-only event log on the same persistent volume as the chat logs; folded into the git-tracked
# overrides.json nightly. Open POST (editing is intentionally open) but guarded by the caps below.
CORRECTIONS_PATH = LOG_PATH.parent / "coauthorship_corrections.jsonl"
CORRECTION_TYPES = {  # event type -> required payload keys
    "node_label": {"id", "label"},
    "node_community": {"id", "community"},
    "node_url": {"id"},
    "node_photo": {"id", "filename"},
    "remove_node": {"id"},
    "paper_rename": {"old", "new"},
    "remove_paper": {"between", "title"},
    "add_paper": {"between", "title"},
    "remove_edge": {"between"},
}
MAX_CORRECTION_BYTES = 4096          # one event's JSON size cap (cheap vandalism brake)
MAX_CORRECTIONS_FILE_BYTES = 8 << 20  # 8 MB total log cap
CORRECTION_RATE = (30, 600)          # at most 30 POSTs per 600s per client IP
_correction_hits: OrderedDict[str, list[float]] = OrderedDict()

# affiliation self-service events (see experiments/coauthorship/affiliation_events.py +
# merge_affiliations.py). Same volume, own log; folded into affiliation_overrides.json nightly.
AFF_EVENTS_PATH = LOG_PATH.parent / "affiliation_events.jsonl"
AFF_EVENT_TYPES = {  # event type -> required payload keys
    "aff_entry_set": {"person", "org", "type"},
    "aff_entry_remove": {"person", "org"},
    "aff_city": {"person", "city"},
    "aff_join": {"name"},
    "aff_confirm": {"person"},
}
MAX_JOIN_EVENT_BYTES = 8192          # joins carry an entries array
_AFF_SEEDS = Path(__file__).parent / "coauthorship" / "seeds.json"

model_client = Cerebras(api_key=CEREBRAS_API_KEY)

# RAG setup
rag_collection = load_collection()

# Pricing for zai-glm-4.7 (USD per token)
INPUT_COST_PER_TOKEN = 2.25 / 1_000_000  # $2.25 per million input tokens
OUTPUT_COST_PER_TOKEN = 2.75 / 1_000_000  # $2.75 per million output tokens
MAX_COST_PER_CONVERSATION = 2.00  # USD

# Server-side conversation store
MAX_CONVERSATIONS = 1000
MAX_TURNS = 50  # max user+assistant message pairs (100 messages total)


@dataclass
class Conversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    cost_usd: float = 0.0


conversations: OrderedDict[str, Conversation] = OrderedDict()


# HTTP requests pydantic models
class ChatRequest(BaseModel):
    user_id: str | None = None
    message: str


class ResetRequest(BaseModel):
    user_id: str


class Correction(BaseModel):
    """One crowd edit to the co-authorship graph. `type` selects the override; `payload` carries it."""
    type: Literal[
        "node_label", "node_community", "node_url", "node_photo", "remove_node",
        "paper_rename", "remove_paper", "add_paper", "remove_edge",
    ]
    payload: dict
    editor: str | None = None
    note: str | None = None


class AffEvent(BaseModel):
    """One affiliation self-service event (edit own row / join the map / confirm own row)."""
    type: Literal["aff_entry_set", "aff_entry_remove", "aff_city", "aff_join", "aff_confirm"]
    payload: dict
    editor: str | None = None
    note: str | None = None


def setup_fastapi_app():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:4000",
            "http://127.0.0.1:4000",
            "http://127.0.0.1:3863",
            "https://alex-loftus.com",
            "https://www.alex-loftus.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

app = setup_fastapi_app()


def get_or_create_conversation(user_id: str | None) -> tuple[str, Conversation]:
    """Look up or create a conversation for user_id. Returns (user_id, Conversation)."""
    if user_id is None:
        user_id = str(uuid.uuid4())

    # Move to end (most recently used)
    if user_id in conversations:
        conversations.move_to_end(user_id)
    else:
        # Evict oldest if at capacity
        while len(conversations) >= MAX_CONVERSATIONS:
            conversations.popitem(last=False)
        conversations[user_id] = Conversation()

    return user_id, conversations[user_id]


def build_messages(
    conversation: list[dict[str, str]],
    rag_context: str | None = None,
) -> list[dict[str, str]]:
    msgs = [SYSTEM_MESSAGE]
    if rag_context:
        msgs.append({"role": "system", "content": rag_context})
    return msgs + conversation


def write_log_entry(
    user_message: str,
    bot_response: str,
    token_latency_ms: float,
    message_count: int,
    token_count: int,
    user_id: str | None,
    conversation_cost_usd: float = 0.0,
    rag_chunks_count: int = 0,
    rag_sources: list[str] | None = None,
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
        "conversation_cost_usd": round(conversation_cost_usd, 6),
        "rag_chunks_count": rag_chunks_count,
        "rag_sources": rag_sources or [],
    }
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"logging failed with error:\n\n {e}")


@app.post("/chat")
def chat_stream(request: ChatRequest, logging: bool = True) -> StreamingResponse:
    user_id, convo = get_or_create_conversation(request.user_id)

    # Rate limit: reject if conversation has already exceeded cost budget
    if convo.cost_usd >= MAX_COST_PER_CONVERSATION:
        raise HTTPException(
            status_code=429,
            detail="Conversation budget exceeded. Please start a new conversation.",
        )

    # Enforce max turns: trim oldest user+assistant pairs if over limit
    while len(convo.messages) >= MAX_TURNS * 2:
        convo.messages.pop(0)  # remove oldest user msg
        if convo.messages:
            convo.messages.pop(0)  # remove oldest assistant msg

    convo.messages.append({"role": "user", "content": request.message})

    # RAG: retrieve relevant chunks for the user's message
    rag_chunks = retrieve_context(rag_collection, request.message)
    rag_text = format_rag_context(rag_chunks)
    messages_for_llm = build_messages(convo.messages, rag_context=rag_text)

    def token_stream():
        try:
            completion = model_client.chat.completions.create(
                model=MODEL,
                messages=messages_for_llm,
                stream=True,
            )
        except Exception as e:
            print(f"LLM API error: {type(e).__name__}: {e}")
            yield f"[Error: LLM API call failed — {type(e).__name__}]"
            return

        bot_response = ""
        completion_tokens = -1
        prompt_tokens = -1
        start = time.time()
        chunk_count = 0
        try:
            for chunk in completion:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    bot_response += delta.content
                    chunk_count += 1
                    yield delta.content
                if hasattr(chunk, "usage") and chunk.usage:
                    if chunk.usage.completion_tokens:
                        completion_tokens = chunk.usage.completion_tokens
                    if chunk.usage.prompt_tokens:
                        prompt_tokens = chunk.usage.prompt_tokens
        except Exception as e:
            print(f"LLM streaming error: {type(e).__name__}: {e}")
            yield f"\n[Error: streaming interrupted — {type(e).__name__}]"
            return

        if completion_tokens == -1:
            completion_tokens = chunk_count
        latency = (time.time() - start) * 1000

        # Update conversation cost from actual token usage
        if prompt_tokens > 0 and completion_tokens > 0:
            turn_cost = (
                prompt_tokens * INPUT_COST_PER_TOKEN
                + completion_tokens * OUTPUT_COST_PER_TOKEN
            )
            convo.cost_usd += turn_cost

        # Store assistant reply in conversation
        convo.messages.append({"role": "assistant", "content": bot_response})

        message_count = len(convo.messages)
        token_latency_ms = (
            latency / completion_tokens if completion_tokens not in [0, -1] else -1
        )
        if logging:
            write_log_entry(
                user_message=request.message,
                bot_response=bot_response,
                token_latency_ms=token_latency_ms,
                message_count=message_count,
                token_count=completion_tokens,
                user_id=user_id,
                conversation_cost_usd=convo.cost_usd,
                rag_chunks_count=len(rag_chunks),
                rag_sources=[c["source"] for c in rag_chunks],
            )

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/reset")
def reset_conversation(request: ResetRequest) -> dict[str, str]:
    """Clear a user's conversation history."""
    conversations.pop(request.user_id, None)
    return {"status": "ok"}


# Pre-warm: load the embedding model on startup to avoid cold-start latency
@app.on_event("startup")
def warmup_rag():
    retrieve_context(rag_collection, "warmup query")
    print(f"RAG ready: {rag_collection.count()} chunks indexed")


# health check
@app.get("/health")
def health_check() -> dict[str, bool | str | int]:
    return {
        "status": "ok",
        "api_key": bool(CEREBRAS_API_KEY),
        "rag_chunks": rag_collection.count(),
        "timestamp": datetime.utcnow().isoformat(),
    }


# log access
def require_bearer(authorization: str | None) -> None:
    """Gate an endpoint on the shared LOG_ACCESS_TOKEN. Raises 500 if unset, 401 if the header is
    missing/malformed/wrong. Used by the log + correction bulk-export endpoints."""
    if LOG_ACCESS_TOKEN is None:
        raise HTTPException(status_code=500)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401)
    if authorization.removeprefix("Bearer ").strip() != LOG_ACCESS_TOKEN:
        raise HTTPException(status_code=401)


@app.get("/logs/download")
def download_logs(authorization: str | None = Header(None)):
    """
    Authorization: Bearer <token>
    """
    require_bearer(authorization)
    if not LOG_PATH.exists():
        raise HTTPException(status_code=404)
    if LOG_PATH.stat().st_size == 0:
        return Response(content="", media_type="text/plain")

    return FileResponse(LOG_PATH)


# ----- co-authorship graph corrections -------------------------------------------------------

def _rate_ok(client_ip: str) -> bool:
    """Sliding-window per-IP limiter for open correction POSTs (in-memory, resets on restart)."""
    limit, window = CORRECTION_RATE
    now = time.time()
    hits = [t for t in _correction_hits.get(client_ip, []) if now - t < window]
    if len(hits) >= limit:
        _correction_hits[client_ip] = hits
        return False
    hits.append(now)
    _correction_hits[client_ip] = hits
    while len(_correction_hits) > MAX_CONVERSATIONS:  # bound memory
        _correction_hits.popitem(last=False)
    return True


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_event(path, kind: str, payload: dict, editor: str | None, note: str | None,
                  client_ip: str, max_bytes: int) -> None:
    """Validated append shared by both event families (size caps raise; caller validates payload)."""
    entry = {
        "type": kind,
        "payload": payload,
        "editor": (editor or "")[:120] or None,
        "note": (note or "")[:500] or None,
        "ts": datetime.utcnow().isoformat(),
        "ip": client_ip,
    }
    blob = json.dumps(entry)
    if len(blob.encode()) > max_bytes:
        raise HTTPException(status_code=413, detail="event too large")
    if path.exists() and path.stat().st_size > MAX_CORRECTIONS_FILE_BYTES:
        raise HTTPException(status_code=507, detail="event log full")
    with path.open("a", encoding="utf-8") as f:
        f.write(blob + "\n")


def _serve_log(path):
    if not path.exists() or path.stat().st_size == 0:
        return Response(content="", media_type="text/plain")
    return FileResponse(path)


def _delete_by_ts(path, ts: str) -> dict:
    events = _read_jsonl(path)
    kept = [e for e in events if e.get("ts") != ts]
    removed = len(events) - len(kept)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"no event with ts={ts}")
    path.write_text("".join(json.dumps(e) + "\n" for e in kept))
    return {"ok": True, "removed": removed, "remaining": len(kept)}


def _read_corrections() -> list[dict]:
    return _read_jsonl(CORRECTIONS_PATH)


@app.post("/coauthorship/corrections")
def submit_correction(correction: Correction, request: Request) -> dict:
    """Append one crowd edit to the correction log. Open, but guarded by per-IP rate + size caps."""
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_ok(client_ip):
        raise HTTPException(status_code=429, detail="too many corrections; try again later")

    required = CORRECTION_TYPES[correction.type]
    missing = required - set(correction.payload)
    if missing:
        raise HTTPException(status_code=422, detail=f"payload missing keys: {sorted(missing)}")
    if "between" in correction.payload:
        pair = correction.payload["between"]
        if not (isinstance(pair, list) and len(pair) == 2):
            raise HTTPException(status_code=422, detail="`between` must be a 2-item list")

    _append_event(CORRECTIONS_PATH, correction.type, correction.payload, correction.editor,
                  correction.note, client_ip, MAX_CORRECTION_BYTES)
    return {"ok": True}


@app.get("/coauthorship/overlay")
def correction_overlay() -> dict:
    """Open, read-only: the live merged overlay so the page can show pending edits pre-rebuild."""
    return fold_events(_read_corrections())


@app.get("/coauthorship/corrections")
def export_corrections(authorization: str | None = Header(None)):
    """Bearer-protected raw event log (audit trail; not needed by the nightly merge)."""
    require_bearer(authorization)
    return _serve_log(CORRECTIONS_PATH)


@app.delete("/coauthorship/corrections")
def delete_correction(ts: str, authorization: str | None = Header(None)) -> dict:
    """Bearer-protected admin revert: drop the event(s) with this exact `ts` from the log.

    This is the durable undo for a bad/vandal edit — the nightly merge regenerates overrides.json
    wholesale from this log, so a git revert of overrides.json alone would be re-clobbered. Removing
    the event here fixes both the live overlay and every future bake in one call:

        curl -X DELETE -H "Authorization: Bearer $TOKEN" \\
             "$API/coauthorship/corrections?ts=2026-06-09T12:34:56.789012"
    """
    require_bearer(authorization)
    return _delete_by_ts(CORRECTIONS_PATH, ts)


# ----- affiliation self-service events --------------------------------------------------------

def _read_aff_events() -> list[dict]:
    return _read_jsonl(AFF_EVENTS_PATH)


def _roster_norms() -> set[str]:
    """Normalized roster names from the seeds shipped in the image + already-folded joins."""
    names = set()
    if _AFF_SEEDS.exists():
        names = {norm_person(s["name"]) for s in json.loads(_AFF_SEEDS.read_text())}
    names |= set(fold_aff_events(_read_aff_events())["join"])
    return names


def _check_aff_fields(payload: dict) -> None:
    for field, cap in AFF_FIELD_CAPS.items():
        v = payload.get(field)
        if v is not None and len(str(v)) > cap:
            raise HTTPException(status_code=422, detail=f"`{field}` longer than {cap} chars")


def _check_aff_entry(spec: dict) -> None:
    if not isinstance(spec, dict) or not str(spec.get("org", "")).strip():
        raise HTTPException(status_code=422, detail="entry needs a non-empty `org`")
    if spec.get("type") not in ENTRY_TYPES:
        raise HTTPException(status_code=422, detail=f"entry `type` must be one of {sorted(ENTRY_TYPES)}")
    _check_aff_fields(spec)


@app.post("/affiliations/corrections")
def submit_aff_event(event: AffEvent, request: Request) -> dict:
    """Append one affiliation self-service event. Open (like the graph corrections), guarded by
    the same per-IP rate limit + size caps; full validation so a bad event can't reach the fold."""
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_ok(client_ip):
        raise HTTPException(status_code=429, detail="too many edits; try again later")

    p = event.payload
    missing = AFF_EVENT_TYPES[event.type] - {k for k, v in p.items() if v is not None}
    if missing:
        raise HTTPException(status_code=422, detail=f"payload missing keys: {sorted(missing)}")
    if event.type == "aff_entry_set":
        _check_aff_entry(p)                 # includes the field caps
    else:
        _check_aff_fields(p)
    if event.type == "aff_join":
        entries = p.get("entries") or []
        if not isinstance(entries, list) or len(entries) > 10:
            raise HTTPException(status_code=422, detail="`entries` must be a list of at most 10")
        for spec in entries:
            _check_aff_entry(spec)
        if norm_person(p["name"]) in _roster_norms():
            raise HTTPException(status_code=409,
                                detail="that name is already on the map — edit the row instead")

    cap = MAX_JOIN_EVENT_BYTES if event.type == "aff_join" else MAX_CORRECTION_BYTES
    _append_event(AFF_EVENTS_PATH, event.type, p, event.editor, event.note, client_ip, cap)
    return {"ok": True}


@app.get("/affiliations/overlay")
def aff_overlay() -> dict:
    """Open, read-only: the folded affiliation overlay (live preview + the nightly merge source)."""
    return fold_aff_events(_read_aff_events())


@app.get("/affiliations/corrections")
def export_aff_events(authorization: str | None = Header(None)):
    """Bearer-protected raw event log (audit trail; carries editor + ip)."""
    require_bearer(authorization)
    return _serve_log(AFF_EVENTS_PATH)


@app.delete("/affiliations/corrections")
def delete_aff_event(ts: str, authorization: str | None = Header(None)) -> dict:
    """Bearer-protected admin revert by exact ts — the durable undo (see delete_correction)."""
    require_bearer(authorization)
    return _delete_by_ts(AFF_EVENTS_PATH, ts)
