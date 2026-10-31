import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from app.llm import LLMResult, ScriptedLLM
from app.main import create_app


def sign(body: bytes, secret: str = "app-secret") -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def payload(message_id="wamid.1", phone="201000000001", text="hello"):
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"id": message_id, "from": phone, "type": "text", "text": {"body": text}}
                            ]
                        }
                    }
                ]
            }
        ]
    }


def build(settings):
    sent = []

    async def fake_send(_settings, phone, text):
        sent.append((phone, text))

    app = create_app(settings, llm=ScriptedLLM([LLMResult(text="أهلاً! أقدر أساعدك تحجز.")] * 5), send=fake_send)
    return TestClient(app), sent


def test_verify_challenge(settings):
    client, _ = build(settings)
    ok = client.get("/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "42"})
    assert ok.status_code == 200 and ok.text == "42"
    bad = client.get("/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "42"})
    assert bad.status_code == 403


def test_unsigned_or_forged_webhooks_are_rejected(settings):
    client, sent = build(settings)
    body = json.dumps(payload()).encode()
    assert client.post("/webhook", content=body).status_code == 401
    assert client.post("/webhook", content=body, headers={"X-Hub-Signature-256": sign(body, "wrong")}).status_code == 401
    assert sent == []


def test_signed_message_gets_a_reply_once(settings):
    client, sent = build(settings)
    body = json.dumps(payload()).encode()
    headers = {"X-Hub-Signature-256": sign(body)}
    assert client.post("/webhook", content=body, headers=headers).status_code == 200
    assert client.post("/webhook", content=body, headers=headers).status_code == 200  # Meta retry
    assert sent == [("201000000001", "أهلاً! أقدر أساعدك تحجز.")]


def test_chat_endpoint_is_off_unless_demo_mode(settings):
    client, _ = build(settings)
    assert client.post("/chat", json={"phone": "201000000001", "text": "hi"}).status_code in (404, 405)
