"""FastAPI app: WhatsApp webhook + an optional local /chat endpoint for demos."""

from __future__ import annotations

import json
import logging
import threading

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .agent import BookingAgent
from .config import Settings, load_settings
from .db import Database
from .llm import LLM, GeminiLLM
from .whatsapp import mask_phone, parse_text_messages, send_text, valid_signature

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clinic-agent")


class ChatIn(BaseModel):
    phone: str = Field(min_length=6, max_length=20, pattern=r"^\+?\d+$")
    text: str = Field(min_length=1, max_length=1000)


def create_app(settings: Settings | None = None, llm: LLM | None = None, send=send_text) -> FastAPI:
    settings = settings or load_settings()
    db = Database(settings.db_path)
    db.init()
    agent = BookingAgent(settings, db, llm or GeminiLLM(settings.gemini_api_key, settings.gemini_model))
    # One SQLite connection is shared, so handle one conversation turn at a time.
    # Plenty for a single clinic; move to Postgres + a queue to scale out.
    lock = threading.Lock()

    def handle_locked(phone: str, text: str) -> str:
        with lock:
            return agent.handle(phone, text)

    app = FastAPI(title="Clinic Booking Agent")
    app.state.db = db
    app.state.agent = agent

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/webhook", response_class=PlainTextResponse)
    def verify(request: Request) -> str:
        q = request.query_params
        if (
            settings.wa_verify_token
            and q.get("hub.mode") == "subscribe"
            and q.get("hub.verify_token") == settings.wa_verify_token
        ):
            return q.get("hub.challenge", "")
        raise HTTPException(status_code=403, detail="Verification failed")

    async def process(phone: str, text: str) -> None:
        reply = await run_in_threadpool(handle_locked, phone, text)
        if reply:
            await send(settings, phone, reply)

    @app.post("/webhook")
    async def receive(request: Request, background: BackgroundTasks) -> dict:
        raw = await request.body()
        if not valid_signature(raw, request.headers.get("X-Hub-Signature-256"), settings.wa_app_secret):
            raise HTTPException(status_code=401, detail="Bad signature")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        for msg in parse_text_messages(payload):
            if not db.mark_processed(msg.message_id):
                continue  # duplicate delivery
            log.info("inbound message from %s", mask_phone(msg.phone))
            background.add_task(process, msg.phone, msg.text)
        # Answer fast: Meta retries webhooks that take too long.
        return {"ok": True}

    if settings.demo_mode:

        @app.post("/chat")
        async def chat(body: ChatIn) -> dict:
            reply = await run_in_threadpool(handle_locked, body.phone, body.text)
            return {"reply": reply}

    return app
# Run with: uvicorn app.main:create_app --factory
