"""Chat backends used by the agent loop.

Any object with a ``generate(messages, tools, tool_choice)`` method returning a
:class:`ModelTurn` can drive an agent. Two backends are provided:

* :class:`OpenAIChatBackend` talks to any OpenAI-compatible
  ``/chat/completions`` endpoint (OpenAI, OpenRouter, vLLM, SGLang, ...).
* :class:`ScriptedBackend` replays a fixed list of actions; it is used by the
  tests and by the ``--backend scripted`` smoke runs.
"""
from __future__ import annotations

import copy
import http.client
import json
import logging
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

Message = Dict[str, Any]

RETRYABLE_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text as produced by the model


@dataclass
class ModelTurn:
    """One assistant response."""

    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    message: Optional[Message] = None  # assistant message to append to history
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def history_message(self) -> Message:
        if self.message is not None:
            return copy.deepcopy(self.message)
        message: Message = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return message


class ChatBackend:
    """Interface implemented by every backend."""

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> ModelTurn:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAIChatBackend(ChatBackend):
    """Minimal OpenAI-compatible client based on the standard library.

    ``extra_body`` is merged into every request and can carry provider-specific
    fields such as ``top_k``, ``reasoning_effort`` or ``chat_template_kwargs``.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.url = _chat_completions_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, messages, tools=None, tool_choice="auto"):
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        for key, value in (("temperature", self.temperature), ("top_p", self.top_p),
                           ("max_tokens", self.max_tokens), ("seed", self.seed)):
            if value is not None:
                payload[key] = value
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            payload["parallel_tool_calls"] = False
        payload.update(self.extra_body)
        data = self._post(payload)
        return _parse_completion(data)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                if exc.code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise RuntimeError("HTTP %d from %s: %s" % (exc.code, self.url, detail)) from exc
                reason = "HTTP %d" % exc.code
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout, OSError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError("request to %s failed: %s" % (self.url, exc)) from exc
                reason = str(exc)
            delay = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
            logger.warning("%s; retrying in %.1fs (%d/%d)", reason, delay, attempt + 1, self.max_retries)
            time.sleep(delay)
        raise AssertionError("unreachable")


def _chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/chat/completions") else url + "/chat/completions"


def _parse_completion(data: Dict[str, Any]) -> ModelTurn:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("response contains no choices: %s" % json.dumps(data)[:500])
    choice = choices[0]
    message = dict(choice.get("message") or {})
    message.setdefault("role", "assistant")
    content = message.get("content")
    if isinstance(content, list):  # content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    calls = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        calls.append(ToolCall(id=raw.get("id") or "call_%d" % index,
                              name=function.get("name") or "", arguments=arguments))
    if calls:
        # Some servers omit call IDs; keep the history consistent with ours.
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in calls
        ]
    usage = data.get("usage") or {}
    return ModelTurn(
        text=(content or "").strip(),
        tool_calls=calls,
        message=message,
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


class ScriptedBackend(ChatBackend):
    """Replays ``actions`` (``(tool_name, arguments_dict)`` pairs), then answers.

    When the loop asks for a final answer early (``tool_choice="none"``), the
    backend answers immediately.
    """

    def __init__(self, actions: Sequence[Any], final_answer: str) -> None:
        self.actions = list(actions)
        self.final_answer = final_answer
        self.calls = 0

    def generate(self, messages, tools=None, tool_choice="auto"):
        self.calls += 1
        if tools and tool_choice != "none" and self.actions:
            name, arguments = self.actions.pop(0)
            call = ToolCall(id="call_%d" % self.calls, name=name, arguments=json.dumps(arguments))
            return ModelTurn(text="", tool_calls=[call])
        return ModelTurn(text="Answer: " + self.final_answer)
