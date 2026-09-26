import threading

from conftest import CounterEnv, CounterTask

from expgym.agent import REGIMES
from expgym.envs.base import ToolResult
from expgym.llm import ScriptedBackend
from expgym.poolact import ExplorationGraph, PoolActHooks, SharedCache, run_pool


def test_cache_respects_simulated_time():
    cache = SharedCache()
    cache.put("probe", {"x": 1}, ToolResult("value=2", 100.0, 2.0), completed_at=100.0)
    assert cache.get("probe", {"x": 1}, now=50.0) is None  # not finished yet for this agent
    assert cache.get("probe", {"x": 1}, now=100.0).text == "value=2"
    cache.put("probe", {"x": 1}, ToolResult("late", 100.0), completed_at=400.0)
    assert cache.get("probe", {"x": 1}, now=100.0).text == "value=2"  # earliest completion wins


def test_graph_shows_pending_requests_of_other_agents():
    env = CounterEnv()
    graph = ExplorationGraph(2, env)
    claim = graph.claim(0, "probe", {"x": 3}, now=0.0)
    assert "probe" in graph.render(viewer=1, now=0.0)
    assert graph.render(viewer=0, now=0.0) == ""  # own requests are not listed
    graph.record(0, "probe", {"x": 3}, ToolResult("value=6", 100.0, 6.0), at=100.0)
    graph.complete(claim, 100.0)
    assert "None completed yet" in graph.render(viewer=1, now=50.0)
    later = graph.render(viewer=1, now=100.0)
    assert "== In Progress ==\n  None." in later and "[agents 0]" in later


def test_withheld_request_is_no_longer_pending():
    env = CounterEnv()
    graph = ExplorationGraph(2, env)
    hooks = PoolActHooks(0, SharedCache(), graph, threading.Lock())
    hooks.on_action("probe", {"x": 1}, 250.0)
    hooks.execute(env, "probe", {"x": 1}, now=250.0, limit=300.0)  # withheld: 250 + 100 >= 300
    assert graph.render(viewer=1, now=260.0) == ""


def test_withheld_results_are_not_shared():
    env, cache = CounterEnv(), SharedCache()
    graph = ExplorationGraph(2, env)
    hooks = PoolActHooks(0, cache, graph, threading.Lock())
    hooks.on_action("probe", {"x": 1}, 250.0)
    result, reused = hooks.execute(env, "probe", {"x": 1}, now=250.0, limit=300.0)
    assert not reused and result.cost == 100.0
    assert len(cache) == 0 and "value" not in graph.render(viewer=1, now=1e9)


def _pool(strategy, regime="free"):
    def backend(env, index):
        return ScriptedBackend([("probe", {"x": 1}), ("probe", {"x": 2})], "42")

    return run_pool(CounterTask(), "0", backend, strategy, REGIMES[regime], n_agents=3)


def test_naive_pool_pays_for_every_call():
    result = _pool("naive")
    assert [a["feedback_cost"] for a in result["agents"]] == [200.0, 200.0, 200.0]
    assert result["aggregate"]["correct"] == 1.0 and result["aggregate"]["method"] == "majority_vote"


def test_poolact_reuses_completed_results():
    env, cache, lock = CounterEnv(), SharedCache(), threading.Lock()
    graph = ExplorationGraph(2, env)
    first, second = (PoolActHooks(i, cache, graph, lock) for i in range(2))
    first.on_action("probe", {"x": 1}, 0.0)
    assert first.execute(env, "probe", {"x": 1}, now=0.0, limit=None) == (ToolResult("value=2", 100.0, 2.0), False)
    early, reused = second.execute(env, "probe", {"x": 1}, now=50.0, limit=None)
    assert not reused and early.cost == 100.0  # result not yet visible at t=50
    late, reused = second.execute(env, "probe", {"x": 1}, now=150.0, limit=None)
    assert reused and late.cost == 0.0


def test_poolact_agents_see_shared_state():
    result = _pool("poolact")
    prompts = [a["messages"][1]["content"] for a in result["agents"]]
    assert all(p.startswith("Probe numbers") for p in prompts)
    # Only the first agent to decide starts without seeing another agent's pending request.
    assert sum("[Parallel Exploration" in p for p in prompts) >= 2
    assert result["cache_entries"] == 2


def test_cached_pool_runs():
    result = _pool("cached", regime="tight")
    assert len(result["agents"]) == 3
    assert all(a["budget"] == 300.0 for a in result["agents"])
