"""Chat with the agent in your terminal, no WhatsApp needed.

    python chat.py               # Gemini first, Groq backup (keys in .env)
    python chat.py --phone 2010...  # pretend to be a different patient
    python chat.py --verbose        # also show retries and provider switches
"""

from __future__ import annotations

import argparse
import logging

from app.agent import BookingAgent
from app.config import load_settings
from app.db import Database
from app.llm import build_llm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone", default="201000000001", help="Patient phone number for this session")
    parser.add_argument("--verbose", action="store_true", help="Show retries and provider fallbacks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.verbose else logging.CRITICAL)
    settings = load_settings()
    db = Database(settings.db_path)
    db.init()
    agent = BookingAgent(settings, db, build_llm(settings))

    print(f"{settings.clinic_name} booking agent. Type 'exit' to quit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if text:
            print(f"Agent: {agent.handle(args.phone, text)}\n")


if __name__ == "__main__":
    main()
