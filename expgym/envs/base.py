"""Environment interface shared by all ExpGym tasks.

An :class:`Environment` is one task instance (one tuning run, one question,
one document). It exposes function-calling tools, charges a simulated cost for
every call, and scores the agent's final answer. A :class:`Task` enumerates
instances and builds a fresh environment for each trajectory.

To add your own environment, subclass both and register the task with
:func:`expgym.envs.register_task` (see ``examples/custom_environment.py``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


class ToolInputError(ValueError):
    """The model supplied invalid tool arguments.

    The message is returned to the model as the observation, at zero cost.
    """


@dataclass
class ToolResult:
    text: str
    """Observation shown to the model."""
    cost: float
    """Simulated feedback cost in seconds."""
    perf: Optional[float] = None
    """Optional numeric outcome (validation performance for tuning)."""


@dataclass
class Observation:
    """A tool call whose result the agent actually saw."""

    name: str
    arguments: Dict[str, Any]
    result: ToolResult


def stable_uniform(key: str, low: float, high: float) -> float:
    """Deterministic pseudo-random number in ``[low, high]`` derived from ``key``."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return low + (high - low) * int.from_bytes(digest[:4], "big") / 2 ** 32


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def function_tool(name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Build an OpenAI-style function tool definition."""
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": parameters}}


class Environment:
    """One task instance. Subclasses implement the abstract methods below."""

    #: Task-specific reference cost ``c_base``; the budget is ``beta * c_base``.
    reference_cost: float = 300.0

    def task_prompt(self) -> str:
        raise NotImplementedError

    def tools(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def call(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute one tool call. Raise :class:`ToolInputError` for bad input."""
        raise NotImplementedError

    def score(self, answer: Optional[str], observations: Sequence[Observation]) -> Dict[str, Any]:
        """Score the final answer. ``observations`` are the results the agent saw."""
        raise NotImplementedError

    def on_reuse(self, name: str, arguments: Dict[str, Any], result: ToolResult) -> None:
        """Called when a result is reused from another agent instead of executed here."""

    # -- Hooks used by PoolAct to render the shared exploration graph. --------

    def describe_action(self, name: str, arguments: Dict[str, Any]) -> str:
        return "%s %s" % (name, canonical_json(arguments))

    def describe_outcome(self, name: str, arguments: Dict[str, Any], result: ToolResult) -> str:
        return self.describe_action(name, arguments)

    def coverage_note(self, outcomes: Sequence[Observation]) -> str:
        return "%d distinct actions completed. Select an action not listed above." % len(outcomes)


class Task:
    """A collection of environment instances."""

    name: str = ""
    #: Aggregation of pool answers: "mean" (tuning) or "vote" (search / audit).
    pool_aggregation: str = "vote"

    def items(self) -> List[str]:
        raise NotImplementedError

    def make_env(self, item: str, repeat: int = 0) -> Environment:
        raise NotImplementedError

    def item_group(self, item: str) -> str:
        """Sub-family used when averaging scores (e.g. ``whois`` / ``whatis``)."""
        return self.name

    def primary_metric(self) -> str:
        raise NotImplementedError

    def vote(self, answers: Sequence[Optional[str]]) -> Optional[str]:
        """Combine the final answers of several agents into one answer."""
        raise NotImplementedError
