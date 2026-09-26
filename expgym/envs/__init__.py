"""Task registry."""
from __future__ import annotations

from typing import Callable, Dict

from expgym.envs.base import Environment, Observation, Task, ToolInputError, ToolResult

_FACTORIES: Dict[str, Callable[[], Task]] = {}


def register_task(name: str, factory: Callable[[], Task]) -> None:
    """Make a task available to the command-line runners under ``name``."""
    _FACTORIES[name] = factory


def get_task(name: str) -> Task:
    if name not in _FACTORIES:
        raise KeyError("unknown task %r; available: %s" % (name, ", ".join(sorted(_FACTORIES))))
    return _FACTORIES[name]()


def available_tasks():
    return sorted(_FACTORIES)


def _tuning():
    from expgym.envs.tuning import TuningTask
    return TuningTask()


def _toy():
    from expgym.envs.tuning import ToyTuningTask
    return ToyTuningTask()


def _search():
    from expgym.envs.search import SearchTask
    return SearchTask()


def _audit():
    from expgym.envs.audit import AuditTask
    return AuditTask()


register_task("tuning", _tuning)
register_task("search", _search)
register_task("audit", _audit)
register_task("toy", _toy)

__all__ = ["Environment", "Observation", "Task", "ToolInputError", "ToolResult",
           "register_task", "get_task", "available_tasks"]
