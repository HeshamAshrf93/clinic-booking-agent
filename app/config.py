"""Settings, read once from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    # LLM
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    # Tried in order when the main model is overloaded (HTTP 429/5xx). Comma-separated.
    gemini_fallback_models: list[str] = field(
        default_factory=lambda: [
            m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-latest").split(",") if m.strip()
        ]
    )

    # Backup provider: any OpenAI-compatible API. Defaults to Groq (free tier).
    # Used when every Gemini model is busy or down. Leave the key empty to disable.
    backup_api_key: str = field(default_factory=lambda: os.getenv("BACKUP_API_KEY", ""))
    backup_base_url: str = field(
        default_factory=lambda: os.getenv("BACKUP_BASE_URL", "https://api.groq.com/openai/v1")
    )
    backup_model: str = field(default_factory=lambda: os.getenv("BACKUP_MODEL", "openai/gpt-oss-120b"))

    # Speech-to-text for voice messages (Whisper). Reuses the backup provider's key by default.
    stt_api_key: str = field(default_factory=lambda: os.getenv("STT_API_KEY") or os.getenv("BACKUP_API_KEY", ""))
    stt_base_url: str = field(
        default_factory=lambda: os.getenv("STT_BASE_URL") or os.getenv("BACKUP_BASE_URL", "https://api.groq.com/openai/v1")
    )
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "whisper-large-v3-turbo"))

    # Storage
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "clinic.db"))

    # Clinic
    clinic_name: str = field(default_factory=lambda: os.getenv("CLINIC_NAME", "Nile Care Clinic"))
    timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo(os.getenv("CLINIC_TZ", "Africa/Cairo")))

    # WhatsApp Cloud API
    wa_access_token: str = field(default_factory=lambda: os.getenv("WA_ACCESS_TOKEN", ""))
    wa_phone_number_id: str = field(default_factory=lambda: os.getenv("WA_PHONE_NUMBER_ID", ""))
    wa_verify_token: str = field(default_factory=lambda: os.getenv("WA_VERIFY_TOKEN", ""))
    wa_app_secret: str = field(default_factory=lambda: os.getenv("WA_APP_SECRET", ""))
    graph_api_version: str = field(default_factory=lambda: os.getenv("GRAPH_API_VERSION", "v21.0"))

    # Safety limits
    max_agent_steps: int = field(default_factory=lambda: int(os.getenv("MAX_AGENT_STEPS", "6")))
    max_messages_per_hour: int = field(default_factory=lambda: int(os.getenv("MAX_MESSAGES_PER_HOUR", "30")))

    # Local demo endpoint (/chat). Off by default: it lets the caller pick any phone number.
    demo_mode: bool = field(default_factory=lambda: _env_bool("DEMO_MODE"))


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return Settings()
