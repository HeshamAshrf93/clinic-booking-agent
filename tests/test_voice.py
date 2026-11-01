"""Voice messages: transcription request, validation, and the /voice demo endpoint."""

from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.llm import LLMResult, ScriptedLLM
from app.main import create_app
from app.voice import TranscriptionError, transcribe


def fake_whisper(text="عايز أحجز عند دكتورة الأسنان", status=200, seen=None):
    def post(url, files, data, headers, timeout):
        if seen is not None:
            seen.update(url=url, files=files, data=data, headers=headers)
        return httpx.Response(status, json={"text": text}, request=httpx.Request("POST", url))

    return post


def test_transcribe_sends_whisper_request(settings):
    s = replace(settings, stt_api_key="key", stt_base_url="https://api.groq.com/openai/v1")
    seen = {}
    assert transcribe(s, b"x" * 3000, "audio/webm;codecs=opus", post=fake_whisper(seen=seen)) == "عايز أحجز عند دكتورة الأسنان"
    assert seen["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert seen["data"]["model"] == "whisper-large-v3-turbo"
    assert seen["files"]["file"][0] == "voice.webm"
    assert "language" not in seen["data"]  # Whisper auto-detects Arabic or English


@pytest.mark.parametrize(
    "key,audio,ctype",
    [("", b"x" * 3000, "audio/webm"), ("k", b"", "audio/webm"), ("k", b"x" * 3000, "video/mp4"), ("k", b"x" * (6 * 1024 * 1024), "audio/webm")],
)
def test_transcribe_rejects_bad_input(settings, key, audio, ctype):
    with pytest.raises(TranscriptionError):
        transcribe(replace(settings, stt_api_key=key), audio, ctype, post=fake_whisper())


def test_voice_endpoint_transcribes_then_runs_the_agent(settings, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "transcribe", lambda s, audio, ctype: "عايز أحجز")
    app = create_app(replace(settings, demo_mode=True), llm=ScriptedLLM([LLMResult(text="أكيد، يوم إيه؟")]))
    client = TestClient(app)
    r = client.post("/voice", params={"phone": "201000000001"}, content=b"x" * 3000, headers={"Content-Type": "audio/webm"})
    assert r.json() == {"transcript": "عايز أحجز", "reply": "أكيد، يوم إيه؟"}
    assert client.post("/voice", params={"phone": "abc"}, content=b"x", headers={"Content-Type": "audio/webm"}).status_code == 422


def test_voice_endpoint_is_off_unless_demo_mode(settings):
    client = TestClient(create_app(settings, llm=ScriptedLLM([])))
    assert client.post("/voice", params={"phone": "201000000001"}, content=b"x").status_code in (404, 405)
