"""Agent loop tests with a scripted LLM, so they run offline and deterministically."""

from app.agent import FALLBACK_REPLY, RATE_LIMIT_REPLY, BookingAgent
from app.llm import LLMResult, ScriptedLLM, ToolCall

PHONE = "201000000001"


def make_agent(settings, db, now, script):
    llm = ScriptedLLM(script)
    return BookingAgent(settings, db, llm, now=now), llm


def test_full_booking_flow_executes_tools_and_feeds_results_back(settings, db, now):
    agent, llm = make_agent(
        settings,
        db,
        now,
        [
            LLMResult(tool_calls=[ToolCall("find_available_slots", {"doctor_id": 1, "date": "2026-11-02"}, id="c1")]),
            LLMResult(text="متاح الساعة 4 العصر. أأكد الحجز باسم مين؟"),
            LLMResult(
                tool_calls=[
                    ToolCall(
                        "book_appointment",
                        {"doctor_id": 1, "start_at": "2026-11-02T16:00:00+02:00", "patient_name": "Alice Samir"},
                        id="c2",
                    )
                ]
            ),
            LLMResult(text="تمام، اتحجزلك يوم الاتنين الساعة 4 مع د. منى حسن."),
        ],
    )

    first = agent.handle(PHONE, "عايز أحجز عند دكتورة الأسنان يوم الاتنين")
    assert first.startswith("متاح")
    tool_msg = llm.seen[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["result"]["ok"]

    second = agent.handle(PHONE, "أيوه باسم Alice Samir")
    assert "اتحجزلك" in second
    rows = db.query("SELECT patient_phone, status FROM appointments")
    assert [(r["patient_phone"], r["status"]) for r in rows] == [(PHONE, "booked")]

    # History is remembered across turns
    assert [m["role"] for m in llm.seen[2][:3]] == ["user", "assistant", "user"]


def test_tool_errors_go_back_to_the_model(settings, db, now):
    agent, llm = make_agent(
        settings,
        db,
        now,
        [
            LLMResult(tool_calls=[ToolCall("book_appointment", {"doctor_id": 1, "start_at": "2026-11-02T03:00:00+02:00", "patient_name": "Ali"})]),
            LLMResult(text="الوقت ده مش متاح، تحب أشوفلك مواعيد تانية؟"),
        ],
    )
    reply = agent.handle(PHONE, "احجزلي الساعة 3 الفجر")
    assert "مش متاح" in reply
    assert llm.seen[1][-1]["result"]["ok"] is False
    assert db.query("SELECT COUNT(*) AS n FROM appointments")[0]["n"] == 0


def test_runaway_loop_stops_and_falls_back(settings, db, now):
    looping = [LLMResult(tool_calls=[ToolCall("list_doctors", {})]) for _ in range(20)]
    agent, llm = make_agent(settings, db, now, looping)
    assert agent.handle(PHONE, "hi") == FALLBACK_REPLY
    assert len(llm.seen) == settings.max_agent_steps


def test_llm_crash_gives_fallback_not_a_stack_trace(settings, db, now):
    class Broken:
        def complete(self, *a, **k):
            raise RuntimeError("API down")

    agent = BookingAgent(settings, db, Broken(), now=now)
    assert agent.handle(PHONE, "hello") == FALLBACK_REPLY


def test_rate_limit_per_phone(settings, db, now):
    agent, _ = make_agent(settings, db, now, [LLMResult(text="ok")] * 10)
    for _ in range(settings.max_messages_per_hour):
        agent.handle(PHONE, "hi")
    assert agent.handle(PHONE, "hi") == RATE_LIMIT_REPLY
    assert agent.handle("201000000009", "hi") == "ok"  # other patients unaffected
