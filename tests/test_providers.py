"""Request formats sent to each provider, and switching between providers."""

import json

import pytest
from google.genai import types

from app.llm import ChainLLM, GeminiLLM, LLMResult, OpenAICompatLLM, ProviderError, ToolCall

TOOLS = [{"name": "list_doctors", "description": "d", "parameters": {"type": "object", "properties": {}}}]
TURN = [
    {"role": "user", "text": "book"},
    {
        "role": "assistant",
        "text": "",
        "tool_calls": [
            {"id": "a", "name": "list_doctors", "args": {}},
            {"id": "b", "name": "find_available_slots", "args": {"doctor_id": 1, "date": "2026-11-02"}},
        ],
    },
    {"role": "tool", "id": "a", "name": "list_doctors", "result": {"ok": True}},
    {"role": "tool", "id": "b", "name": "find_available_slots", "result": {"ok": True, "slots": []}},
]


def test_gemini_sends_tool_results_as_one_user_turn():
    llm = GeminiLLM("k", "m", sleep=lambda s: None)
    sent = {}

    def fake_generate(model, contents, config):
        sent["contents"] = contents
        c = types.Content(role="model", parts=[types.Part.from_text(text="ok")])
        return types.GenerateContentResponse(candidates=[types.Candidate(content=c)])

    llm._generate = fake_generate
    llm.complete("sys", TURN, TOOLS)
    roles = [c.role for c in sent["contents"]]
    assert roles == ["user", "model", "user"]  # Gemini rejects a "tool" role
    assert len(sent["contents"][2].parts) == 2
    assert all(p.function_response for p in sent["contents"][2].parts)


def test_openai_compatible_request_and_tool_call_parsing():
    seen = {}

    def fake_post(url, payload):
        seen["url"], seen["payload"] = url, payload
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "c1", "type": "function",
                             "function": {"name": "find_available_slots", "arguments": '{"doctor_id": 1, "date": "2026-11-02"}'}}
                        ],
                    }
                }
            ]
        }

    llm = OpenAICompatLLM("k", "llama", "https://api.groq.com/openai/v1", post=fake_post)
    result = llm.complete("sys", TURN, TOOLS)

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    msgs = seen["payload"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "tool"]
    assert json.loads(msgs[2]["tool_calls"][1]["function"]["arguments"]) == {"doctor_id": 1, "date": "2026-11-02"}
    assert msgs[3]["tool_call_id"] == "a"
    assert seen["payload"]["tools"][0]["function"]["name"] == "list_doctors"
    assert result.tool_calls[0].name == "find_available_slots"
    assert result.tool_calls[0].args == {"doctor_id": 1, "date": "2026-11-02"}


def test_openai_compatible_retries_then_raises():
    calls = []

    def busy(url, payload):
        calls.append(1)
        raise ProviderError(503, "busy")

    llm = OpenAICompatLLM("k", "m", "https://x/v1", retries=1, sleep=lambda s: None, post=busy)
    with pytest.raises(ProviderError):
        llm.complete("sys", [{"role": "user", "text": "hi"}], TOOLS)
    assert len(calls) == 2


class Fake:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def complete(self, system, messages, tools):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_chain_switches_to_backup_when_gemini_is_down():
    gemini = Fake([RuntimeError("503 busy")])
    backup = Fake([LLMResult(tool_calls=[ToolCall("list_doctors", {}, "x")]), LLMResult(text="done")])
    chain = ChainLLM([("gemini", gemini), ("backup", backup)])

    chain.complete("s", [{"role": "user", "text": "hi"}], TOOLS)
    assert chain.last_provider == "backup"
    # Mid-turn calls stay on the backup; Gemini is not retried within the turn.
    chain.complete("s", [{"role": "user", "text": "hi"}, {"role": "tool", "id": "x", "name": "list_doctors", "result": {}}], TOOLS)
    assert gemini.calls == 1 and backup.calls == 2


def test_chain_tries_gemini_again_on_the_next_turn():
    gemini = Fake([RuntimeError("down"), LLMResult(text="back")])
    backup = Fake([LLMResult(text="from backup")])
    chain = ChainLLM([("gemini", gemini), ("backup", backup)])
    chain.complete("s", [{"role": "user", "text": "1"}], TOOLS)
    assert chain.complete("s", [{"role": "user", "text": "2"}], TOOLS).text == "back"


def test_chain_needs_at_least_one_provider():
    with pytest.raises(RuntimeError):
        ChainLLM([])


def test_with_a_backup_gemini_gets_one_quick_try():
    from app.config import Settings
    from app.llm import build_llm

    chain = build_llm(Settings(gemini_api_key="g", backup_api_key="b"))
    gemini = dict(chain._providers)["gemini"]
    assert gemini._retries == 0 and gemini._models == ["gemini-3.8-flash"]
    assert [name for name, _ in chain._providers] == ["gemini", "backup"]


def test_no_tools_means_no_tools_field():
    seen = {}

    def fake_post(url, payload):
        seen.update(payload)
        return {"choices": [{"message": {"content": "hi"}}]}

    OpenAICompatLLM("k", "m", "https://x/v1", post=fake_post).complete("s", [{"role": "user", "text": "x"}], [])
    assert "tools" not in seen and "tool_choice" not in seen

    llm = GeminiLLM("k", "m", sleep=lambda s: None)
    configs = []

    def fake_generate(model, contents, config):
        configs.append(config)
        c = types.Content(role="model", parts=[types.Part.from_text(text="ok")])
        return types.GenerateContentResponse(candidates=[types.Candidate(content=c)])

    llm._generate = fake_generate
    llm.complete("s", [{"role": "user", "text": "x"}], [])
    assert not configs[0].tools


def test_provider_order_is_configurable():
    from app.config import Settings
    from app.llm import build_llm

    chain = build_llm(Settings(gemini_api_key="g", backup_api_key="b", provider_order=["backup", "gemini"]))
    assert [name for name, _ in chain._providers] == ["backup", "gemini"]
    only_gemini = build_llm(Settings(gemini_api_key="g", backup_api_key="", provider_order=["backup", "gemini"]))
    assert [name for name, _ in only_gemini._providers] == ["gemini"]


def test_gemini_thinking_level_is_sent():
    llm = GeminiLLM("k", "m", sleep=lambda s: None, thinking="low")
    configs = []

    def fake_generate(model, contents, config):
        configs.append(config)
        c = types.Content(role="model", parts=[types.Part.from_text(text="ok")])
        return types.GenerateContentResponse(candidates=[types.Candidate(content=c)])

    llm._generate = fake_generate
    llm.complete("s", [{"role": "user", "text": "x"}], [])
    assert configs[0].thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_rate_limit_waits_for_the_time_groq_asks_for():
    waits, calls = [], []

    def post(url, payload):
        calls.append(1)
        if len(calls) == 1:
            raise ProviderError(429, '{"error":{"message":"Rate limit reached ... Please try again in 3.2s."}}')
        return {"choices": [{"message": {"content": "ok"}}]}

    llm = OpenAICompatLLM("k", "m", "https://x/v1", sleep=waits.append, post=post)
    assert llm.complete("s", [{"role": "user", "text": "x"}], []).text == "ok"
    assert waits == [3.7]
