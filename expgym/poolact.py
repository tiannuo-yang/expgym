"""Multi-agent strategies: Naive, Cached and PoolAct.

All strategies run ``N`` agents on the same task instance, each with its own
feedback budget and step limit.

* **naive** - independent rollouts.
* **cached** - agents share completed tool results. Re-issuing a request that
  another agent has already completed returns its result at no cost.
* **poolact** - the shared cache plus a shared exploration graph (completed
  results, exploration paths and pending requests) that is shown to each agent
  before it acts. Action selection is serialized by a lock covering context
  injection, the model's decision and registration of the chosen request; the
  lock is released before the tool runs, so tool execution stays parallel.

Agents live on a simulated clock: an agent's clock is its cumulative feedback
cost. A shared result becomes visible to an agent once the agent's clock has
passed the result's completion time, so agents cannot see results "from the
future". Results withheld at a budget boundary are never shared.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from expgym.agent import AgentHooks, Regime, run_agent
from expgym.envs.base import Environment, Observation, Task, ToolResult, canonical_json
from expgym.llm import ChatBackend

STRATEGIES = ("naive", "cached", "poolact")


class SharedCache:
    """Completed tool results keyed by tool name and canonical arguments."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[Tuple[str, str], Tuple[ToolResult, float]] = {}
        self.hits = 0

    def get(self, name: str, arguments: Dict[str, Any], now: float) -> Optional[ToolResult]:
        with self._lock:
            entry = self._entries.get((name, canonical_json(arguments)))
            if entry is None or entry[1] > now:
                return None
            self.hits += 1
            return entry[0]

    def put(self, name: str, arguments: Dict[str, Any], result: ToolResult, completed_at: float) -> None:
        key = (name, canonical_json(arguments))
        with self._lock:
            if key not in self._entries or completed_at < self._entries[key][1]:
                self._entries[key] = (result, completed_at)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class _Event:
    agent: int
    key: str
    outcome: Observation
    completed_at: float


@dataclass
class _Claim:
    agent: int
    key: str
    label: str
    started_at: float
    completed_at: float = float("inf")


class ExplorationGraph:
    """Shared record of completed and in-progress actions across a pool."""

    def __init__(self, n_agents: int, env: Environment) -> None:
        self.n_agents = n_agents
        self.env = env  # used only to render actions and coverage
        self._lock = threading.Lock()
        self._events: List[_Event] = []
        self._claims: List[_Claim] = []

    def claim(self, agent: int, name: str, arguments: Dict[str, Any], now: float) -> _Claim:
        claim = _Claim(agent, _key(name, arguments), self.env.describe_action(name, arguments), now)
        with self._lock:
            self._claims.append(claim)
        return claim

    def complete(self, claim: _Claim, at: float) -> None:
        with self._lock:
            claim.completed_at = max(claim.started_at, at)

    def record(self, agent: int, name: str, arguments: Dict[str, Any], result: ToolResult, at: float) -> None:
        with self._lock:
            self._events.append(_Event(agent, _key(name, arguments), Observation(name, arguments, result), at))

    def close_agent(self, agent: int, at: float) -> None:
        with self._lock:
            for claim in self._claims:
                if claim.agent == agent and claim.completed_at == float("inf"):
                    claim.completed_at = max(claim.started_at, at)

    def render(self, viewer: int, now: float) -> str:
        """The shared state as seen by agent ``viewer`` at simulated time ``now``."""
        with self._lock:
            events = [e for e in self._events if e.completed_at <= now]
            pending = [c for c in self._claims
                       if c.agent != viewer and c.started_at <= now < c.completed_at]
        if not events and not pending:
            return ""
        lines = ["[Parallel Exploration — Agent %d of %d]" % (viewer, self.n_agents),
                 "You are one of %d agents solving this task in parallel." % self.n_agents,
                 "Shared state below shows what other agents have explored and are exploring.",
                 "Use this to plan your next action — prioritize paths not yet explored.",
                 "", "== In Progress =="]
        lines += ["  %s [agent %d]" % (c.label, c.agent) for c in sorted(pending, key=lambda c: c.started_at)] \
            or ["  None."]

        nodes: Dict[str, Tuple[Observation, set]] = {}
        edges: Dict[Tuple[str, str], set] = {}
        edge_counts: Dict[Tuple[str, str], int] = {}
        last: Dict[int, str] = {}
        for event in events:
            if event.key not in nodes:
                nodes[event.key] = (event.outcome, set())
            nodes[event.key][1].add(event.agent)
            if event.agent in last:
                edge = (last[event.agent], event.key)
                edges.setdefault(edge, set()).add(event.agent)
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
            last[event.agent] = event.key
        labels = {key: self.env.describe_outcome(o.name, o.arguments, o.result) for key, (o, _) in nodes.items()}

        lines += ["", "== Already Explored =="]
        ordered = sorted(nodes.items(), key=lambda item: -(item[1][0].result.perf or 0.0))
        for key, (_, agents) in ordered:
            lines.append("  %s [agents %s]%s" % (labels[key], _agents(agents), " (you)" if viewer in agents else ""))
        if not nodes:
            lines.append("  None completed yet.")

        lines += ["", "== Exploration Paths =="]
        for edge, agents in sorted(edges.items(), key=lambda item: -edge_counts[item[0]]):
            count = edge_counts[edge]
            lines.append("  %s --> %s [%sagents %s]" % (labels[edge[0]], labels[edge[1]],
                                                       "%dx, " % count if count > 1 else "", _agents(agents)))
        if not edges:
            lines.append("  No multi-step paths recorded yet.")

        lines += ["", "== Coverage Gap ==",
                  "  " + self.env.coverage_note([outcome for outcome, _ in nodes.values()])]
        return "\n".join(lines)


def _key(name: str, arguments: Dict[str, Any]) -> str:
    return name + " " + canonical_json(arguments)


def _agents(agents) -> str:
    return ",".join(str(a) for a in sorted(agents))


class CachedHooks(AgentHooks):
    """Reuse completed results from the shared cache."""

    def __init__(self, cache: SharedCache) -> None:
        self.cache = cache

    def execute(self, env, name, arguments, now, limit):
        hit = self.cache.get(name, arguments, now)
        if hit is not None:
            env.on_reuse(name, arguments, hit)
            return ToolResult(hit.text, 0.0, hit.perf), True
        result = env.call(name, arguments)
        done = now + result.cost
        if limit is None or done < limit:  # withheld results are not shared
            self.cache.put(name, arguments, result, done)
        return result, False


class PoolActHooks(CachedHooks):
    """Shared cache + exploration graph + decision lock for one agent."""

    def __init__(self, agent: int, cache: SharedCache, graph: ExplorationGraph,
                 lock: Optional[threading.Lock]) -> None:
        super().__init__(cache)
        self.agent = agent
        self.graph = graph
        self.lock = lock
        self._claim: Optional[_Claim] = None

    @contextmanager
    def decision(self) -> Iterator[None]:
        if self.lock is None:
            yield
        else:
            with self.lock:
                yield

    def context_note(self, now):
        return self.graph.render(self.agent, now)

    def on_action(self, name, arguments, now):
        self._claim = self.graph.claim(self.agent, name, arguments, now)

    def execute(self, env, name, arguments, now, limit):
        done = now  # a failed or withheld request is simply no longer pending
        try:
            result, reused = super().execute(env, name, arguments, now, limit)
            if reused or limit is None or now + result.cost < limit:
                done = now + result.cost
                self.graph.record(self.agent, name, arguments, result, done)
            return result, reused
        finally:
            if self._claim is not None:
                self.graph.complete(self._claim, done)
                self._claim = None

    def on_finish(self, now):
        self.graph.close_agent(self.agent, now)


def run_pool(
    task: Task,
    item: str,
    make_backend: Callable[[Environment, int], ChatBackend],
    strategy: str,
    regime: Regime,
    *,
    n_agents: int = 4,
    repeat: int = 0,
    use_lock: bool = True,
    **agent_kwargs: Any,
) -> Dict[str, Any]:
    """Run ``n_agents`` agents on one task instance and aggregate their answers.

    ``make_backend(env, agent_index)`` builds each agent's backend (e.g. with a
    distinct sampling seed). ``use_lock=False`` runs PoolAct without the
    decision lock (the lock ablation).
    """
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be one of %s" % (STRATEGIES,))
    envs = [task.make_env(item, repeat) for _ in range(n_agents)]
    cache = SharedCache()
    graph = ExplorationGraph(n_agents, envs[0])
    lock = threading.Lock() if use_lock else None

    def hooks_for(agent: int) -> AgentHooks:
        if strategy == "naive":
            return AgentHooks()
        if strategy == "cached":
            return CachedHooks(cache)
        return PoolActHooks(agent, cache, graph, lock)

    def run(agent: int) -> Dict[str, Any]:
        return run_agent(make_backend(envs[agent], agent), envs[agent], regime, hooks=hooks_for(agent), **agent_kwargs)

    with ThreadPoolExecutor(max_workers=n_agents) as pool:
        agents = list(pool.map(run, range(n_agents)))

    return {"strategy": strategy, "n_agents": n_agents, "aggregate": aggregate(task, envs[0], agents),
            "cache_entries": len(cache), "cache_hits": cache.hits, "agents": agents}


def aggregate(task: Task, env: Environment, agents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool score: mean of agent scores (tuning) or the score of the voted answer."""
    if task.pool_aggregation == "mean":
        metric = task.primary_metric()
        scores = [a["score"][metric] for a in agents]
        return {"method": "mean_individual", metric: sum(scores) / len(scores),
                "best_of_n_" + metric: max(scores)}
    answer = task.vote([a["answer"] for a in agents])
    return dict(method="majority_vote", answer=answer, **env.score(answer, []))
