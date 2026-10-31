"""SQLite storage: doctors, their weekly schedule, appointments and chat history."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    id          INTEGER PRIMARY KEY,
    name_en     TEXT NOT NULL,
    name_ar     TEXT NOT NULL,
    specialty   TEXT NOT NULL,
    fee_egp     INTEGER NOT NULL
);

-- One row per working block. weekday: 0 = Monday ... 6 = Sunday (Python's convention).
CREATE TABLE IF NOT EXISTS schedules (
    doctor_id   INTEGER NOT NULL REFERENCES doctors(id),
    weekday     INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),
    start_time  TEXT NOT NULL,   -- HH:MM
    end_time    TEXT NOT NULL,   -- HH:MM
    slot_minutes INTEGER NOT NULL DEFAULT 30
);

CREATE TABLE IF NOT EXISTS appointments (
    id            INTEGER PRIMARY KEY,
    doctor_id     INTEGER NOT NULL REFERENCES doctors(id),
    start_at      TEXT NOT NULL,          -- ISO 8601, clinic local time with offset
    patient_phone TEXT NOT NULL,
    patient_name  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'booked' CHECK (status IN ('booked', 'cancelled')),
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- The database itself refuses a second active booking for the same doctor and time,
-- even if two messages race each other.
CREATE UNIQUE INDEX IF NOT EXISTS one_booking_per_slot
    ON appointments (doctor_id, start_at) WHERE status = 'booked';

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    phone       TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS messages_by_phone ON messages (phone, id);

CREATE TABLE IF NOT EXISTS handoffs (
    id          INTEGER PRIMARY KEY,
    phone       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- WhatsApp may deliver the same webhook more than once.
CREATE TABLE IF NOT EXISTS processed_messages (
    message_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SEED_DOCTORS = [
    (1, "Dr. Mona Hassan", "د. منى حسن", "Dentistry", 400),
    (2, "Dr. Karim Adel", "د. كريم عادل", "Physiotherapy", 350),
    (3, "Dr. Sara Fathy", "د. سارة فتحي", "Dermatology", 500),
]

# Clinic is closed on Friday (weekday 4).
SEED_SCHEDULES = [
    # Dentistry: Sat, Mon, Wed evenings
    (1, 5, "16:00", "21:00", 30), (1, 0, "16:00", "21:00", 30), (1, 2, "16:00", "21:00", 30),
    # Physiotherapy: Sun, Tue, Thu, 45-minute sessions
    (2, 6, "10:00", "16:00", 45), (2, 1, "10:00", "16:00", 45), (2, 3, "10:00", "16:00", 45),
    # Dermatology: Sat and Tue
    (3, 5, "12:00", "17:00", 20), (3, 1, "17:00", "21:00", 20),
]


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        # A single shared connection keeps ":memory:" databases alive for tests.
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def init(self, seed: bool = True) -> None:
        self._conn.executescript(SCHEMA)
        if seed and not self.query("SELECT 1 FROM doctors LIMIT 1"):
            with self.tx() as c:
                c.executemany("INSERT INTO doctors VALUES (?, ?, ?, ?, ?)", SEED_DOCTORS)
                c.executemany(
                    "INSERT INTO schedules (doctor_id, weekday, start_time, end_time, slot_minutes)"
                    " VALUES (?, ?, ?, ?, ?)",
                    SEED_SCHEDULES,
                )

    # ---- chat history -------------------------------------------------
    def add_message(self, phone: str, role: str, text: str) -> None:
        self.execute("INSERT INTO messages (phone, role, text) VALUES (?, ?, ?)", (phone, role, text))

    def recent_messages(self, phone: str, limit: int = 12) -> list[dict]:
        rows = self.query(
            "SELECT role, text FROM messages WHERE phone = ? ORDER BY id DESC LIMIT ?", (phone, limit)
        )
        return [{"role": r["role"], "text": r["text"]} for r in reversed(rows)]

    def user_messages_last_hour(self, phone: str) -> int:
        row = self.query(
            "SELECT COUNT(*) AS n FROM messages WHERE phone = ? AND role = 'user'"
            " AND created_at >= datetime('now', '-1 hour')",
            (phone,),
        )[0]
        return int(row["n"])

    def mark_processed(self, message_id: str) -> bool:
        """Return True the first time a WhatsApp message id is seen, False on a retry."""
        try:
            self.execute("INSERT INTO processed_messages (message_id) VALUES (?)", (message_id,))
            return True
        except sqlite3.IntegrityError:
            return False
