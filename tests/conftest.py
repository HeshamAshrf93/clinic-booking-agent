from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.config import Settings
from app.db import Database

CAIRO = ZoneInfo("Africa/Cairo")
# Sunday 1 Nov 2026, 09:00 Cairo. Monday 2 Nov is a dentistry day (16:00-21:00).
FIXED_NOW = datetime(2026, 11, 1, 9, 0, tzinfo=CAIRO)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        gemini_api_key="test",
        db_path=":memory:",
        wa_verify_token="verify-me",
        wa_app_secret="app-secret",
        max_messages_per_hour=5,
    )


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    database.init()
    return database


@pytest.fixture
def now():
    return lambda: FIXED_NOW
