"""LLM providers behind one small interface, so the agent loop is provider-neutral.

Messages use a plain format:
    {"role": "user", "text": ...}
    {"role": "assistant", "text": ..., "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "id": ..., "name": ..., "result": {...}}
"""

from __future__ import annotations

import json
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
        timeout_s: int = 30,
    ) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        from google import genai  # imported lazily so tests don't need the SDK configured

        from google.genai import types

        # Fail fast instead of hanging when the API is slow; the chain then tries the backup.
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))
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
                # Gemini expects tool results in a "user" turn; results of parallel calls
                # from the same model turn go together in one Content.
                if contents and contents[-1].role == "user" and contents[-1].parts and contents[-1].parts[0].function_response:
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))

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
            ]
            if tools
            else None,
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


class ProviderError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.code = status


class OpenAICompatLLM:
    """Any OpenAI-compatible chat API with tool calling: Groq, OpenAI, OpenRouter...

    Uses plain HTTP so it needs no extra SDK. The default target is Groq's free tier.
    """

    RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        retries: int = 1,
        sleep: Callable[[float], None] = time.sleep,
        post: Callable[..., Any] | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("Backup API key is not set.")
        self._key = api_key
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._retries = retries
        self._sleep = sleep
        self._post = post  # injectable for tests

    def _to_messages(self, system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc.get("id") or f"call_{i}",
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"], ensure_ascii=False)},
                        }
                        for i, tc in enumerate(m["tool_calls"])
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.get("id") or "call_0",
                        "content": json.dumps(m["result"], ensure_ascii=False),
                    }
                )
        return out

    def _request(self, payload: dict) -> dict:
        if self._post is not None:
            return self._post(self._url, payload)
        import httpx

        try:
            resp = httpx.post(
                self._url, json=payload, headers={"Authorization": f"Bearer {self._key}"}, timeout=30
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(504, "timeout") from exc
        if resp.status_code >= 400:
            raise ProviderError(resp.status_code, resp.text[:200])
        return resp.json()

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResult:
        payload = {
            "model": self._model,
            "temperature": 0.2,
            "messages": self._to_messages(system, messages),
        }
        if tools:  # APIs reject an empty tools list
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            payload["tool_choice"] = "auto"
        for attempt in range(self._retries + 1):
            try:
                data = self._request(payload)
                break
            except ProviderError as exc:
                if exc.code not in self.RETRYABLE or attempt == self._retries:
                    raise
                self._sleep(2**attempt)

        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}  # the tool layer rejects missing arguments safely
            calls.append(ToolCall(name=tc["function"]["name"], args=args, id=tc.get("id")))
        return LLMResult(text=(msg.get("content") or "").strip() if not calls else "", tool_calls=calls)


class ChainLLM:
    """Tries providers in order. A conversation turn sticks to the provider that started it,
    because tool-call ids and Gemini thought signatures don't transfer between providers."""

    def __init__(self, providers: list[tuple[str, LLM]]) -> None:
        if not providers:
            raise RuntimeError("No LLM configured. Set GEMINI_API_KEY and/or BACKUP_API_KEY in .env.")
        self._providers = providers
        self._active: tuple[str, LLM] | None = None
        self.last_provider: str | None = None

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResult:
        new_turn = bool(messages) and messages[-1]["role"] == "user"
        candidates = self._providers if (new_turn or self._active is None) else [self._active]
        last_error: Exception | None = None
        for name, provider in candidates:
            try:
                result = provider.complete(system, messages, tools)
                self._active = (name, provider)
                self.last_provider = name
                return result
            except Exception as exc:  # noqa: BLE001 - any provider failure moves to the next one
                last_error = exc
                log.warning("provider %s failed: %s", name, str(exc)[:160])
        assert last_error is not None
        raise last_error


def build_llm(settings) -> LLM:
    """Gemini first, the OpenAI-compatible backup (Groq by default) second."""
    providers: list[tuple[str, LLM]] = []
    has_backup = bool(settings.backup_api_key)
    if settings.gemini_api_key:
        if has_backup:
            # A backup exists: give Gemini one quick try, then hand over, so the patient
            # waits seconds, not a minute of retries.
            gemini = GeminiLLM(settings.gemini_api_key, settings.gemini_model, [], retries=0, timeout_s=15)
        else:
            gemini = GeminiLLM(settings.gemini_api_key, settings.gemini_model, settings.gemini_fallback_models)
        providers.append(("gemini", gemini))
    if has_backup:
        providers.append(
            ("backup", OpenAICompatLLM(settings.backup_api_key, settings.backup_model, settings.backup_base_url))
        )
    return ChainLLM(providers)


class ScriptedLLM:
    """Deterministic fake used in tests and offline demos: replays a list of LLMResults."""

    def __init__(self, script: list[LLMResult]) -> None:
        self.script = list(script)
        self.seen: list[list[dict]] = []
        self.systems: list[str] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResult:
        self.seen.append([dict(m) for m in messages])
        self.systems.append(system)
        if not self.script:
            return LLMResult(text="(no more scripted replies)")
        return self.script.pop(0)
