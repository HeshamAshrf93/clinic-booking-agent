"""LLM providers behind one small interface, so the agent loop is provider-neutral.

Messages use a plain format:
    {"role": "user", "text": ...}
    {"role": "assistant", "text": ..., "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "id": ..., "name": ..., "result": {...}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


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

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        from google import genai  # imported lazily so tests don't need the SDK configured

        self._client = genai.Client(api_key=api_key)
        self._model = model

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
        response = self._client.models.generate_content(model=self._model, contents=contents, config=config)

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
