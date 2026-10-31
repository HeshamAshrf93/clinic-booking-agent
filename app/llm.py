"""LLM providers behind one small interface, so the agent loop is provider-neutral.

Messages use a plain format:
    {"role": "user", "text": ...}
    {"role": "assistant", "text": ..., "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "id": ..., "name": ..., "result": {...}}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class LLMResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider-specific turn, replayed as-is within the same request


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResult: ...


class GeminiLLM:
    """Google Gemini with native function calling (free tier works for development)."""

    RETRYABLE = {429, 500, 503, 504}  # rate limited, overloaded or temporarily down

    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_models: list[str] | None = None,
        retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        from google import genai  # imported lazily so tests don't need the SDK configured

        self._client = genai.Client(api_key=api_key)
        self._models = [model] + [m for m in (fallback_models or []) if m and m != model]
        self._retries = retries
        self._sleep = sleep
        self._turn_model: str | None = None

    def _generate(self, model: str, contents, config):
        return self._client.models.generate_content(model=model, contents=contents, config=config)

    def _call_with_resilience(self, contents, config, new_turn: bool):
        """Retry busy errors with backoff, then fall back to the next model.

        A function-calling turn must stay on one model (its thought signatures are model
        specific), so the fallback choice is made on the first call of a turn and kept.
        """
        from google.genai import errors

        candidates = self._models if (new_turn or self._turn_model is None) else [self._turn_model]
        last_error: Exception | None = None
        for model in candidates:
            for attempt in range(self._retries + 1):
                try:
                    response = self._generate(model, contents, config)
                    self._turn_model = model
                    return response
                except errors.APIError as exc:
                    last_error = exc
                    if exc.code not in self.RETRYABLE:
                        raise
                    if attempt < self._retries:
                        self._sleep(2**attempt)  # 1s, 2s
                    log.warning("Gemini %s busy (HTTP %s), attempt %d", model, exc.code, attempt + 1)
        assert last_error is not None
        raise last_error

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResult:
        from google.genai import types

        contents: list[types.Content] = []
        for m in messages:
            if m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m["text"])]))
            elif m["role"] == "assistant":
                if m.get("_raw") is not None:
                    # Send the model's own turn back untouched: it carries thought signatures
                    # that Gemini needs to continue a function-calling turn.
                    contents.append(m["_raw"])
                    continue
                parts = []
                if m.get("text"):
                    parts.append(types.Part.from_text(text=m["text"]))
                for tc in m.get("tool_calls", []):
                    parts.append(types.Part(function_call=types.FunctionCall(id=tc.get("id"), name=tc["name"], args=tc["args"])))
                contents.append(types.Content(role="model", parts=parts))
            elif m["role"] == "tool":
                part = types.Part(
                    function_response=types.FunctionResponse(id=m.get("id"), name=m["name"], response=m["result"])
                )
                contents.append(types.Content(role="tool", parts=[part]))

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
                        )
                        for t in tools
                    ]
                )
            ],
            # We run the tools ourselves so every call goes through our validation.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        new_turn = bool(messages) and messages[-1]["role"] == "user"
        response = self._call_with_resilience(contents, config, new_turn)

        calls = [ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id) for fc in (response.function_calls or [])]
        text = ""
        if not calls:
            try:
                text = response.text or ""
            except ValueError:  # blocked or empty candidate
                text = ""
        raw = response.candidates[0].content if response.candidates else None
        return LLMResult(text=text.strip(), tool_calls=calls, raw=raw)


class ScriptedLLM:
    """Deterministic fake used in tests and offline demos: replays a list of LLMResults."""

    def __init__(self, script: list[LLMResult]) -> None:
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResult:
        self.seen.append([dict(m) for m in messages])
        if not self.script:
            return LLMResult(text="(no more scripted replies)")
        return self.script.pop(0)
