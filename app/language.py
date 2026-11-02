"""Decide the reply language in code, from the patient's latest message only."""

from __future__ import annotations

import re

_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]")

NAMES = {"ar": "Egyptian Arabic", "en": "English"}


def detect_language(text: str) -> str | None:
    """Return "ar", "en", or None if there are no words.

    Counts words, not letters, and leans Arabic for mixed messages: Egyptian patients often
    write Arabic with English names or terms ("أيوه باسم Alice Samir", "عايز appointment").
    """
    words = (text or "").split()
    arabic = sum(1 for w in words if _ARABIC.search(w))
    latin = sum(1 for w in words if _LATIN.search(w) and not _ARABIC.search(w))
    if arabic == 0 and latin == 0:
        return None
    return "ar" if arabic and arabic * 2 >= latin else "en"


SHORT_MESSAGE_WORDS = 4


def reply_language(latest: str, earlier_user_messages: list[str]) -> str | None:
    """Language to reply in.

    Normally the latest message decides. Short messages (a name, "ok", a time) don't carry a
    reliable signal, so they keep the conversation's language: "Hesham Ashraf" in the middle of
    an Arabic chat is still an Arabic chat.
    """
    lang = detect_language(latest)
    if len((latest or "").split()) > SHORT_MESSAGE_WORDS:
        return lang
    for earlier in reversed(earlier_user_messages):
        if len(earlier.split()) > SHORT_MESSAGE_WORDS and (prev := detect_language(earlier)):
            return prev
    return lang
