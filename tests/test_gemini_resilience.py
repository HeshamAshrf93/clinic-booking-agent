"""GeminiLLM retries busy errors, falls back to another model, and keeps one model per turn."""

import pytest
from google.genai import errors, types

from app.llm import GeminiLLM


def busy(code=503):
    return errors.ServerError(code, {"error": {"code": code, "message": "busy", "status": "UNAVAILABLE"}})


def text_response(text="ok"):
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def make_llm(outcomes, fallbacks=("backup-model",)):
    """outcomes: dict model -> list of exceptions/responses returned in order."""
    llm = GeminiLLM("test-key", "main-model", list(fallbacks), retries=2, sleep=lambda s: None)
    calls = []

    def fake_generate(model, contents, config):
        calls.append(model)
        result = outcomes[model].pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    llm._generate = fake_generate
    return llm, calls


USER = [{"role": "user", "text": "hi"}]


def test_retries_then_succeeds_on_same_model():
    llm, calls = make_llm({"main-model": [busy(), busy(429), text_response("hello")]})
    assert llm.complete("sys", USER, []).text == "hello"
    assert calls == ["main-model"] * 3


def test_falls_back_when_main_model_stays_busy():
    llm, calls = make_llm({"main-model": [busy()] * 3, "backup-model": [text_response("from backup")]})
    assert llm.complete("sys", USER, []).text == "from backup"
    assert calls == ["main-model"] * 3 + ["backup-model"]


def test_rest_of_turn_stays_on_the_model_that_answered():
    llm, calls = make_llm({"main-model": [busy()] * 3, "backup-model": [text_response("a"), text_response("b")]})
    llm.complete("sys", USER, [])
    mid_turn = USER + [{"role": "tool", "id": "1", "name": "list_doctors", "result": {"ok": True}}]
    llm.complete("sys", mid_turn, [])
    assert calls[-1] == "backup-model"


def test_non_retryable_errors_are_raised_immediately():
    bad_request = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    llm, calls = make_llm({"main-model": [bad_request]})
    with pytest.raises(errors.ClientError):
        llm.complete("sys", USER, [])
    assert calls == ["main-model"]


def test_gives_up_after_all_models_are_busy():
    llm, calls = make_llm({"main-model": [busy()] * 3, "backup-model": [busy()] * 3})
    with pytest.raises(errors.ServerError):
        llm.complete("sys", USER, [])
    assert len(calls) == 6
