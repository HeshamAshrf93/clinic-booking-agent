"""Business rules live in code. These tests prove the model can't talk its way around them."""

from app.tools import ClinicTools

from .conftest import CAIRO

ALICE = "201000000001"
BOB = "201000000002"
MONDAY_4PM = "2026-11-02T16:00:00+02:00"


def tools_for(db, phone, now):
    return ClinicTools(db, phone, CAIRO, now=now)


def test_slots_follow_the_doctors_schedule(db, now):
    result = tools_for(db, ALICE, now).find_available_slots(1, "2026-11-02")
    assert result["ok"]
    assert result["slots"][0] == {"start_at": MONDAY_4PM, "time": "16:00"}
    assert len(result["slots"]) == 8 and result["more_available"]


def test_day_off_returns_no_slots(db, now):
    result = tools_for(db, ALICE, now).find_available_slots(1, "2026-11-06")  # Friday
    assert result["ok"] and result["slots"] == []


def test_past_and_far_dates_rejected(db, now):
    t = tools_for(db, ALICE, now)
    assert not t.find_available_slots(1, "2026-10-01")["ok"]
    assert not t.find_available_slots(1, "2027-06-01")["ok"]
    assert not t.find_available_slots(1, "next monday")["ok"]


def test_booking_removes_slot_and_blocks_double_booking(db, now):
    alice, bob = tools_for(db, ALICE, now), tools_for(db, BOB, now)
    booked = alice.book_appointment(1, MONDAY_4PM, "Alice Samir")
    assert booked["ok"]

    starts = [s["start_at"] for s in bob.find_available_slots(1, "2026-11-02")["slots"]]
    assert MONDAY_4PM not in starts

    again = bob.book_appointment(1, MONDAY_4PM, "Bob Nabil")
    assert not again["ok"] and "taken" in again["error"]


def test_cannot_book_invented_or_past_times(db, now):
    t = tools_for(db, ALICE, now)
    assert not t.book_appointment(1, "2026-11-02T16:10:00+02:00", "Alice Samir")["ok"]  # not a slot
    assert not t.book_appointment(1, "2026-11-02T03:00:00+02:00", "Alice Samir")["ok"]  # clinic closed
    assert not t.book_appointment(1, "2026-10-26T16:00:00+02:00", "Alice Samir")["ok"]  # past
    assert not t.book_appointment(99, MONDAY_4PM, "Alice Samir")["ok"]  # no such doctor
    assert not t.book_appointment(1, MONDAY_4PM, "")["ok"]  # no name


def test_limit_of_active_bookings_per_patient(db, now):
    t = tools_for(db, ALICE, now)
    assert t.book_appointment(1, "2026-11-02T16:00:00+02:00", "Alice Samir")["ok"]
    assert t.book_appointment(1, "2026-11-02T16:30:00+02:00", "Alice Samir")["ok"]
    third = t.book_appointment(1, "2026-11-02T17:00:00+02:00", "Alice Samir")
    assert not third["ok"]


def test_patient_cannot_see_or_cancel_someone_elses_booking(db, now):
    alice, bob = tools_for(db, ALICE, now), tools_for(db, BOB, now)
    appt_id = alice.book_appointment(1, MONDAY_4PM, "Alice Samir")["appointment"]["appointment_id"]

    assert bob.list_my_appointments()["appointments"] == []
    attempt = bob.cancel_appointment(appt_id)
    assert not attempt["ok"]
    assert len(alice.list_my_appointments()["appointments"]) == 1

    assert alice.cancel_appointment(appt_id)["ok"]
    assert alice.list_my_appointments()["appointments"] == []
    # The slot is free again
    starts = [s["start_at"] for s in bob.find_available_slots(1, "2026-11-02")["slots"]]
    assert MONDAY_4PM in starts


def test_unknown_tool_and_bad_args_fail_safely(db, now):
    t = tools_for(db, ALICE, now)
    assert not t.call("drop_database", {})["ok"]
    assert not t.call("book_appointment", {"doctor_id": 1})["ok"]
    assert not t.call("cancel_appointment", {"appointment_id": "abc"})["ok"]
