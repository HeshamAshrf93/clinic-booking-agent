"""The tools the agent can call.

Every rule that matters (is the slot real? is it free? is it in the future? whose booking is
this?) is enforced here in code, never left to the model. The patient's phone number comes
from the WhatsApp session, not from the model's arguments, so the model cannot book or cancel
on someone else's behalf even if a message tries to trick it.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .db import Database

MAX_DAYS_AHEAD = 30
MAX_ACTIVE_BOOKINGS = 2
MAX_SLOTS_RETURNED = 8

# JSON-schema descriptions the LLM sees. Kept provider-neutral.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_doctors",
        "description": "List the clinic's doctors with their specialty, fee in EGP and working days.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "find_available_slots",
        "description": (
            "Find free appointment times for one doctor on one date. Always call this before "
            "offering or booking a time; never invent times."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doctor_id": {"type": "integer", "description": "Doctor id from list_doctors."},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD (clinic local time)."},
            },
            "required": ["doctor_id", "date"],
        },
    },
    {
        "name": "find_next_available",
        "description": (
            "Find the next days a doctor has free times, searching up to 14 days ahead in one call. "
            "Use this when the patient asks for the earliest time, any day, or a day the doctor doesn't work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doctor_id": {"type": "integer", "description": "Doctor id from list_doctors."},
                "from_date": {"type": "string", "description": "Start searching from this date, YYYY-MM-DD. Defaults to today."},
            },
            "required": ["doctor_id"],
        },
    },
    {
        "name": "book_appointment",
        "description": (
            "Book a slot returned by find_available_slots, for the patient you are chatting with. "
            "Only call after the patient has confirmed the doctor, date, time and their name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doctor_id": {"type": "integer"},
                "start_at": {
                    "type": "string",
                    "description": "Exact start_at value of the chosen slot from find_available_slots or find_next_available.",
                },
                "patient_name": {"type": "string", "description": "Patient's full name."},
            },
            "required": ["doctor_id", "start_at", "patient_name"],
        },
    },
    {
        "name": "list_my_appointments",
        "description": "List the upcoming appointments of the patient you are chatting with.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel_appointment",
        "description": "Cancel one of the patient's own upcoming appointments, after they confirm.",
        "parameters": {
            "type": "object",
            "properties": {"appointment_id": {"type": "integer"}},
            "required": ["appointment_id"],
        },
    },
    {
        "name": "request_human",
        "description": (
            "Hand the conversation to clinic staff. Use for medical questions, complaints, "
            "emergencies, payments, or anything outside booking."
        ),
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


class ClinicTools:
    """Tools bound to one patient (the phone number of the current chat)."""

    def __init__(
        self,
        db: Database,
        patient_phone: str,
        tz: ZoneInfo,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.db = db
        self.phone = patient_phone
        self.tz = tz
        self._now = now or (lambda: datetime.now(tz))

    def now(self) -> datetime:
        return self._now().astimezone(self.tz)

    # ---- dispatcher ---------------------------------------------------
    def call(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        handlers = {
            "list_doctors": self.list_doctors,
            "find_available_slots": self.find_available_slots,
            "find_next_available": self.find_next_available,
            "book_appointment": self.book_appointment,
            "list_my_appointments": self.list_my_appointments,
            "cancel_appointment": self.cancel_appointment,
            "request_human": self.request_human,
        }
        handler = handlers.get(name)
        if handler is None:
            return _err(f"Unknown tool: {name}")
        try:
            return handler(**(args or {}))
        except TypeError as exc:  # wrong or missing arguments from the model
            return _err(f"Invalid arguments for {name}: {exc}")

    # ---- helpers ------------------------------------------------------
    def _doctor(self, doctor_id: Any) -> sqlite3.Row | None:
        try:
            doctor_id = int(doctor_id)
        except (TypeError, ValueError):
            return None
        rows = self.db.query("SELECT * FROM doctors WHERE id = ?", (doctor_id,))
        return rows[0] if rows else None

    def _slots_for_day(self, doctor_id: int, day: date) -> list[datetime]:
        blocks = self.db.query(
            "SELECT start_time, end_time, slot_minutes FROM schedules WHERE doctor_id = ? AND weekday = ?",
            (doctor_id, day.weekday()),
        )
        slots: list[datetime] = []
        for b in blocks:
            start = datetime.combine(day, time.fromisoformat(b["start_time"]), self.tz)
            end = datetime.combine(day, time.fromisoformat(b["end_time"]), self.tz)
            step = timedelta(minutes=b["slot_minutes"])
            t = start
            while t + step <= end:
                slots.append(t)
                t += step
        return sorted(slots)

    def _booked_starts(self, doctor_id: int, day: date) -> set[str]:
        rows = self.db.query(
            "SELECT start_at FROM appointments WHERE doctor_id = ? AND status = 'booked' AND start_at LIKE ?",
            (doctor_id, f"{day.isoformat()}%"),
        )
        return {r["start_at"] for r in rows}

    # ---- tools --------------------------------------------------------
    def list_doctors(self) -> dict[str, Any]:
        doctors = []
        for d in self.db.query("SELECT * FROM doctors ORDER BY id"):
            days = self.db.query(
                "SELECT DISTINCT weekday FROM schedules WHERE doctor_id = ? ORDER BY weekday", (d["id"],)
            )
            doctors.append(
                {
                    "doctor_id": d["id"],
                    "name_en": d["name_en"],
                    "name_ar": d["name_ar"],
                    "specialty": d["specialty"],
                    "fee_egp": d["fee_egp"],
                    "working_days": [WEEKDAYS[r["weekday"]] for r in days],
                }
            )
        return {"ok": True, "doctors": doctors}

    def find_available_slots(self, doctor_id: Any, date: str) -> dict[str, Any]:  # noqa: A002
        doctor = self._doctor(doctor_id)
        if doctor is None:
            return _err("No doctor with that id. Call list_doctors first.")
        try:
            day = datetime.strptime(str(date), "%Y-%m-%d").date()
        except ValueError:
            return _err("Date must be YYYY-MM-DD.")
        now = self.now()
        if day < now.date():
            return _err("That date is in the past.")
        if day > now.date() + timedelta(days=MAX_DAYS_AHEAD):
            return _err(f"Bookings open at most {MAX_DAYS_AHEAD} days ahead.")

        all_slots = self._slots_for_day(doctor["id"], day)
        if not all_slots:
            return {"ok": True, "doctor_id": doctor["id"], "date": day.isoformat(), "slots": [],
                    "note": f"{doctor['name_en']} does not work on {WEEKDAYS[day.weekday()]}."}

        booked = self._booked_starts(doctor["id"], day)
        free = [s for s in all_slots if s > now and s.isoformat() not in booked]
        return {
            "ok": True,
            "doctor_id": doctor["id"],
            "date": day.isoformat(),
            "slots": [{"start_at": s.isoformat(), "time": s.strftime("%H:%M")} for s in free[:MAX_SLOTS_RETURNED]],
            "more_available": len(free) > MAX_SLOTS_RETURNED,
        }

    def find_next_available(self, doctor_id: Any, from_date: str | None = None, days: int = 14) -> dict[str, Any]:
        doctor = self._doctor(doctor_id)
        if doctor is None:
            return _err("No doctor with that id. Call list_doctors first.")
        today = self.now().date()
        start = today
        if from_date:
            try:
                start = max(today, datetime.strptime(str(from_date), "%Y-%m-%d").date())
            except ValueError:
                return _err("from_date must be YYYY-MM-DD.")
        found = []
        for offset in range(min(int(days), MAX_DAYS_AHEAD)):
            day = start + timedelta(days=offset)
            if day > today + timedelta(days=MAX_DAYS_AHEAD):
                break
            result = self.find_available_slots(doctor["id"], day.isoformat())
            if result.get("ok") and result["slots"]:
                found.append({"date": day.isoformat(), "day": WEEKDAYS[day.weekday()], "slots": result["slots"][:4]})
            if len(found) == 3:
                break
        return {"ok": True, "doctor_id": doctor["id"], "next_days": found,
                "note": "" if found else "No free times in the next two weeks."}

    def book_appointment(self, doctor_id: Any, start_at: str, patient_name: str) -> dict[str, Any]:
        doctor = self._doctor(doctor_id)
        if doctor is None:
            return _err("No doctor with that id.")
        name = " ".join(str(patient_name or "").split())
        if not (2 <= len(name) <= 60):
            return _err("Ask the patient for their full name first.")
        try:
            start = datetime.fromisoformat(str(start_at))
        except ValueError:
            return _err("start_at must be an exact value from find_available_slots.")
        if start.tzinfo is None:
            start = start.replace(tzinfo=self.tz)
        start = start.astimezone(self.tz)

        if start <= self.now():
            return _err("That time has already passed.")
        if start not in self._slots_for_day(doctor["id"], start.date()):
            return _err("That is not one of the doctor's working slots. Call find_available_slots.")

        upcoming = self._upcoming_for_patient()
        if len(upcoming) >= MAX_ACTIVE_BOOKINGS:
            return _err(
                f"The patient already has {MAX_ACTIVE_BOOKINGS} upcoming appointments. "
                "They must cancel one before booking another."
            )

        try:
            with self.db.tx() as c:
                cur = c.execute(
                    "INSERT INTO appointments (doctor_id, start_at, patient_phone, patient_name)"
                    " VALUES (?, ?, ?, ?)",
                    (doctor["id"], start.isoformat(), self.phone, name),
                )
                appointment_id = cur.lastrowid
        except sqlite3.IntegrityError:
            return _err("Sorry, that slot was just taken. Offer the patient other times.")

        return {
            "ok": True,
            "appointment": {
                "appointment_id": appointment_id,
                "doctor": doctor["name_en"],
                "doctor_ar": doctor["name_ar"],
                "specialty": doctor["specialty"],
                "start_at": start.isoformat(),
                "day": WEEKDAYS[start.weekday()],
                "fee_egp": doctor["fee_egp"],
                "patient_name": name,
            },
        }

    def _upcoming_for_patient(self) -> list[sqlite3.Row]:
        rows = self.db.query(
            "SELECT a.*, d.name_en, d.name_ar, d.specialty FROM appointments a"
            " JOIN doctors d ON d.id = a.doctor_id"
            " WHERE a.patient_phone = ? AND a.status = 'booked' ORDER BY a.start_at",
            (self.phone,),
        )
        now = self.now()
        return [r for r in rows if datetime.fromisoformat(r["start_at"]) > now]

    def list_my_appointments(self) -> dict[str, Any]:
        return {
            "ok": True,
            "appointments": [
                {
                    "appointment_id": r["id"],
                    "doctor": r["name_en"],
                    "doctor_ar": r["name_ar"],
                    "specialty": r["specialty"],
                    "start_at": r["start_at"],
                    "patient_name": r["patient_name"],
                }
                for r in self._upcoming_for_patient()
            ],
        }

    def cancel_appointment(self, appointment_id: Any) -> dict[str, Any]:
        try:
            appointment_id = int(appointment_id)
        except (TypeError, ValueError):
            return _err("appointment_id must be a number from list_my_appointments.")
        mine = {r["id"]: r for r in self._upcoming_for_patient()}
        if appointment_id not in mine:
            # Same answer whether it doesn't exist or belongs to someone else: no information leak.
            return _err("No upcoming appointment with that id for this patient.")
        self.db.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,))
        return {"ok": True, "cancelled_appointment_id": appointment_id, "start_at": mine[appointment_id]["start_at"]}

    def request_human(self, reason: str) -> dict[str, Any]:
        reason = str(reason or "")[:300]
        self.db.execute("INSERT INTO handoffs (phone, reason) VALUES (?, ?)", (self.phone, reason))
        return {"ok": True, "handed_off": True, "note": "Tell the patient a staff member will reply soon."}
