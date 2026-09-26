"""ReAct agent loop with native function calling and a feedback budget.

Each assistant turn either calls exactly one tool or submits a final answer.
Every tool call is charged a simulated cost. With a finite budget ``B``, the
action that brings cumulative cost to or beyond ``B`` is charged but its
observation is withheld, and the agent must answer from earlier observations.
The same forced final answer is requested when the step limit is reached.
"""
from __future__ import annotations

import copy
import json
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from expgym.envs.base import Environment, Observation, ToolInputError, ToolResult
from expgym.llm import ChatBackend, Message, ModelTurn

SYSTEM_PROMPT = (
    "Interaction protocol: use the API's supplied function tools to acquire feedback. "
    "A textual Action directive or an imitation of a tool result is not a tool call. "
    "Make at most one function call per assistant turn. After receiving the real tool "
    "result, decide whether another observation is useful. When ready, submit your "
    "final answer in the task's required format (Answer: followed by the answer is "
    "also accepted). Do not invent observations or claim an unexecuted call succeeded."
)

WITHHELD_NOTICE = (
    "Observation: [over-budget — result withheld. "
    "This evaluation reached or exceeded the time budget.]"
)


@dataclass(frozen=True)
class Regime:
    """A feedback regime: budget ``B = beta * c_base`` and cost visibility."""

    name: str
    beta: float  # math.inf for no spending ceiling
    show_cost: bool

    def limit(self, reference_cost: float) -> Optional[float]:
        return None if math.isinf(self.beta) else self.beta * reference_cost


REGIMES: Dict[str, Regime] = {
    "free": Regime("free", math.inf, show_cost=False),
    "moderate": Regime("moderate", 10.0, show_cost=True),
    "tight": Regime("tight", 3.0, show_cost=True),
}


def get_regime(name: str) -> Regime:
    """``free`` / ``moderate`` / ``tight``, or ``beta=<x>`` for a custom budget."""
    if name in REGIMES:
        return REGIMES[name]
    if name.startswith("beta="):
        beta = float(name.split("=", 1)[1])
        if not beta > 0:
            raise ValueError("beta must be positive")
        return Regime(name, beta, show_cost=True)
    raise ValueError("unknown regime %r; use free, moderate, tight or beta=<x>" % name)


class AgentHooks:
    """Extension points used by multi-agent strategies (see :mod:`expgym.poolact`).

    The defaults implement a single independent agent.
    """

    @contextmanager
    def decision(self) -> Iterator[None]:
        """Context held while the agent reads its context and picks an action."""
        yield

    def context_note(self, now: float) -> str:
        """Text appended to the newest environment message before a decision."""
        return ""

    def on_action(self, name: str, arguments: Dict[str, Any], now: float) -> None:
        """Called (inside :meth:`decision`) once an action has been chosen."""

    def execute(self, env: Environment, name: str, arguments: Dict[str, Any],
                now: float, limit: Optional[float]) -> Tuple[ToolResult, bool]:
        """Run the tool. Returns ``(result, reused)``."""
        return env.call(name, arguments), False

    def on_finish(self, now: float) -> None:
        """Called once when the agent stops."""


def extract_answer(text: str) -> Optional[str]:
    """Return the final answer in ``text``; an ``Answer:`` label is optional."""
    text = re.sub(r"<(think|thinking|reasoning)>.*?</\1>", "", text or "", flags=re.S | re.I)
    labels = list(re.finditer(r"(?:final\s+)?answer\s*[*_]*\s*:\s*[*_]*", text, re.I))
    if labels:
        text = text[labels[-1].end():]
    text = text.strip().strip("*_").strip()
    return text or None


def run_agent(
    backend: ChatBackend,
    env: Environment,
    regime: Regime,
    *,
    max_steps: int = 30,
    max_invalid_turns: int = 1,
    max_context_tokens: Optional[int] = None,
    hooks: Optional[AgentHooks] = None,
) -> Dict[str, Any]:
    """Run one agent on one environment and return its scored trajectory.

    Args:
        max_steps: interaction limit; every model turn counts, including
            zero-cost tool calls and malformed turns.
        max_invalid_turns: malformed turns (no parseable action or answer)
            tolerated before the agent is asked for its final answer.
        max_context_tokens: optional rough client-side context cap; older tool
            outputs are shortened and then dropped to fit.
    """
    hooks = hooks or AgentHooks()
    tools = env.tools()
    tool_names = {tool["function"]["name"] for tool in tools}
    limit = regime.limit(env.reference_cost)
    messages: List[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": env.task_prompt()},
    ]
    spent = 0.0
    answer: Optional[str] = None
    termination = "step_limit"
    invalid_turns = 0
    observations: List[Observation] = []
    call_log: List[Dict[str, Any]] = []
    usage = {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "model_seconds": 0.0}
    annotated: set = set()

    def ask(tool_choice: str) -> ModelTurn:
        index = len(messages) - 1
        note = hooks.context_note(spent)
        if note and index not in annotated and messages[index]["role"] in ("user", "tool"):
            messages[index] = dict(messages[index], content=messages[index]["content"] + "\n\n" + note)
            annotated.add(index)
        started = time.time()
        turn = backend.generate(_fit_context(messages, max_context_tokens), tools, tool_choice)
        usage["model_calls"] += 1
        usage["prompt_tokens"] += turn.prompt_tokens
        usage["completion_tokens"] += turn.completion_tokens
        usage["model_seconds"] += time.time() - started
        messages.append(turn.history_message())
        return turn

    for _ in range(max_steps):
        with hooks.decision():
            turn = ask("auto")
            action, error = _parse_decision(turn, tool_names)
            if error is None and action is None:
                answer = extract_answer(turn.text)
                termination = "answered"
                break
            if error is not None:
                invalid_turns += 1
                _reply_error(messages, turn, error)
                if invalid_turns > max_invalid_turns:
                    termination = "invalid_response"
                    break
                continue
            call_id, name, arguments = action
            hooks.on_action(name, arguments, spent)

        try:
            result, reused = hooks.execute(env, name, arguments, spent, limit)
        except ToolInputError as exc:
            result, reused = ToolResult("Tool error: %s" % exc, 0.0), False
        spent += result.cost
        withheld = limit is not None and spent >= limit
        call_log.append({"name": name, "arguments": arguments, "cost": result.cost,
                         "perf": result.perf, "reused": reused, "visible": not withheld,
                         "cumulative_cost": spent})
        if withheld:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": WITHHELD_NOTICE})
            termination = "budget_exhausted"
            break
        observations.append(Observation(name, arguments, result))
        text = "Observation: " + result.text
        if regime.show_cost and result.cost >= 0.5:
            text += " | cost=%.0fs" % result.cost
            if limit is not None:
                text += " [time_left=%.0fs]" % max(0.0, limit - spent)
        messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    if answer is None:
        reason = {"step_limit": "Maximum steps reached",
                  "budget_exhausted": "Time budget exceeded",
                  "invalid_response": "Invalid response"}[termination]
        messages.append({"role": "user", "content": (
            "System: Loop aborted (%s). Respond immediately with Answer: "
            "<your final choice> and no other text." % reason)})
        with hooks.decision():
            turn = ask("none")
        if not turn.tool_calls and turn.finish_reason != "length":
            answer = extract_answer(turn.text)
    hooks.on_finish(spent)

    return {
        "answer": answer,
        "score": env.score(answer, observations),
        "termination": termination,
        "feedback_cost": spent,
        "budget": limit,
        "steps": usage["model_calls"],
        "tool_calls": call_log,
        "usage": usage,
        "messages": messages,
    }


def _parse_decision(turn: ModelTurn, tool_names: set):
    """Return ``((call_id, name, arguments), None)``, ``(None, None)`` for an
    answer, or ``(None, error_message)`` for a malformed turn."""
    if turn.finish_reason == "length":
        return None, "Model completion token limit reached"
    if not turn.tool_calls:
        if extract_answer(turn.text) is None:
            return None, "Empty response"
        return None, None
    if len(turn.tool_calls) != 1:
        return None, "Exactly one tool call is allowed per turn"
    call = turn.tool_calls[0]
    if call.name not in tool_names:
        return None, "Unknown tool '%s'" % call.name
    try:
        arguments = json.loads(call.arguments or "{}")
    except ValueError as exc:
        return None, "Invalid function arguments: %s" % exc
    if not isinstance(arguments, dict):
        return None, "Function arguments must be a JSON object"
    return (call.id, call.name, arguments), None


def _reply_error(messages: List[Message], turn: ModelTurn, error: str) -> None:
    notice = "Protocol error: %s. No tool was executed." % error
    if turn.tool_calls:
        for call in turn.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": notice})
    else:
        messages.append({"role": "user", "content":
                         notice + " Use the supplied function tools or submit your final answer."})


def _estimate_tokens(messages: List[Message]) -> int:
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 3


def _fit_context(messages: List[Message], max_tokens: Optional[int]) -> List[Message]:
    """Shorten old tool outputs, then drop the oldest turns, to fit ``max_tokens``.

    The system prompt, task prompt and the latest turns are always kept, and
    an assistant tool call is never separated from its result.
    """
    if max_tokens is None or _estimate_tokens(messages) <= max_tokens:
        return messages
    head, rest = messages[:2], copy.deepcopy(messages[2:])
    groups: List[List[Message]] = []
    for message in rest:
        if message["role"] == "assistant" or not groups:
            groups.append([])
        groups[-1].append(message)
    for group in groups[:-2]:
        for message in group:
            content = message.get("content")
            if message["role"] == "tool" and isinstance(content, str) and len(content) > 600:
                message["content"] = content[:600] + "\n[... truncated ...]"
    while len(groups) > 2 and _estimate_tokens(head + sum(groups, [])) > max_tokens:
        groups.pop(0)
    return head + sum(groups, [])
