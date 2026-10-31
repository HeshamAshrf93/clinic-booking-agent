"""WhatsApp Cloud API helpers: verify webhooks, parse inbound messages, send replies."""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class InboundMessage:
    message_id: str
    phone: str
    text: str


def valid_signature(raw_body: bytes, signature_header: str | None, app_secret: str) -> bool:
    """Meta signs every webhook with X-Hub-Signature-256 = 'sha256=' + HMAC(app_secret, body)."""
    if not app_secret or not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.removeprefix("sha256="))


def parse_text_messages(payload: dict[str, Any]) -> list[InboundMessage]:
    """Pull text messages out of a webhook payload; ignore statuses, media, reactions."""
    found: list[InboundMessage] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}
            for msg in value.get("messages", []) or []:
                if msg.get("type") == "text" and msg.get("from") and msg.get("id"):
                    body = (msg.get("text") or {}).get("body", "")
                    found.append(InboundMessage(message_id=msg["id"], phone=msg["from"], text=body))
    return found


def mask_phone(phone: str) -> str:
    return phone[:4] + "****" + phone[-2:] if len(phone) > 6 else "****"


async def send_text(settings: Settings, to_phone: str, text: str) -> None:
    url = f"https://graph.facebook.com/{settings.graph_api_version}/{settings.wa_phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text[:4096]},
    }
    headers = {"Authorization": f"Bearer {settings.wa_access_token}"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload, headers=headers)
    if resp.status_code >= 400:
        # Log the status only: the response may echo the patient's number.
        log.error("WhatsApp send failed for %s: HTTP %s", mask_phone(to_phone), resp.status_code)
