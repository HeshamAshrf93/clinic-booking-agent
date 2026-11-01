"""Speech-to-text with Whisper through an OpenAI-compatible API (Groq by default, free tier).

Used by the browser demo's mic button. The same function can transcribe WhatsApp voice notes.
"""

from __future__ import annotations

from typing import Callable

import httpx

from .config import Settings

MAX_AUDIO_BYTES = 5 * 1024 * 1024  # about 5 minutes of webm/opus; the demo stops at 60 s
EXTENSIONS = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/mpeg": "mp3", "audio/wav": "wav"}


class TranscriptionError(RuntimeError):
    pass


def transcribe(
    settings: Settings,
    audio: bytes,
    content_type: str,
    post: Callable[..., httpx.Response] | None = None,
) -> str:
    if not settings.stt_api_key:
        raise TranscriptionError("Voice needs BACKUP_API_KEY (Groq) in .env.")
    if not audio:
        raise TranscriptionError("Empty recording.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise TranscriptionError("Recording is too long.")

    mime = content_type.split(";")[0].strip().lower()
    ext = EXTENSIONS.get(mime)
    if ext is None:
        raise TranscriptionError(f"Unsupported audio type: {mime}")

    url = settings.stt_base_url.rstrip("/") + "/audio/transcriptions"
    # No language hint: Whisper detects Arabic or English itself, so patients can use either.
    files = {"file": (f"voice.{ext}", audio, mime)}
    data = {"model": settings.stt_model, "response_format": "json", "temperature": "0"}
    headers = {"Authorization": f"Bearer {settings.stt_api_key}"}
    try:
        resp = (post or httpx.post)(url, files=files, data=data, headers=headers, timeout=30)
    except httpx.HTTPError as exc:
        raise TranscriptionError(f"Speech service unreachable: {exc}") from exc
    if resp.status_code >= 400:
        raise TranscriptionError(f"Speech service error HTTP {resp.status_code}")
    return (resp.json().get("text") or "").strip()
