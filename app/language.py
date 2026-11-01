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
