"""Command-line interface: ``expgym run``, ``expgym pool`` and ``expgym summarize``."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from expgym.agent import get_regime, run_agent
from expgym.envs import available_tasks, get_task
from expgym.envs.base import Environment, Task
from expgym.llm import ChatBackend, OpenAIChatBackend, ScriptedBackend
from expgym.poolact import STRATEGIES, run_pool
from expgym.summary import summarize

DEFAULT_REPEATS = {"tuning": 3, "audit": 3}  # expgym run
POOL_REPEATS = {"tuning": 3}                  # expgym pool


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

def add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model backend")
    group.add_argument("--backend", choices=("openai", "scripted"), default="openai",
                       help="'openai': any OpenAI-compatible endpoint; 'scripted': offline smoke test.")
    group.add_argument("--model", default=os.environ.get("EXPGYM_MODEL"), help="Model name sent to the API.")
    group.add_argument("--base-url", default=os.environ.get("EXPGYM_BASE_URL", "https://api.openai.com/v1"))
    group.add_argument("--api-key-env", default="EXPGYM_API_KEY",
                       help="Environment variable holding the API key (default: EXPGYM_API_KEY).")
    group.add_argument("--temperature", type=float)
    group.add_argument("--top-p", type=float)
    group.add_argument("--max-tokens", type=int)
    group.add_argument("--seed", type=int,
                       help="Base sampling seed; every run and every pool agent gets its own offset from it.")
    group.add_argument("--extra-body", type=_json_object, default={},
                       help='JSON merged into every request, e.g. \'{"reasoning_effort": "high"}\'.')
    group.add_argument("--timeout", type=float, default=600.0)
    group.add_argument("--max-retries", type=int, default=5)


def _json_object(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("not valid JSON: %s" % exc)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return value


def make_backend_factory(args: argparse.Namespace) -> Callable[[Environment, int], ChatBackend]:
    if args.backend == "scripted":
        return lambda env, index: scripted_backend(env, seed=index)
    if not args.model:
        raise SystemExit("--model is required (or set EXPGYM_MODEL)")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        logging.warning("$%s is not set; sending requests without an API key", args.api_key_env)

    def factory(env: Environment, index: int) -> ChatBackend:
        return OpenAIChatBackend(
            args.model, args.base_url, api_key, temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_tokens, seed=None if args.seed is None else args.seed + index,
            extra_body=args.extra_body, timeout=args.timeout, max_retries=args.max_retries)
    return factory


def scripted_backend(env: Environment, seed: int = 0, n_actions: int = 3) -> ScriptedBackend:
    """A deterministic stand-in for a model, used to check the pipeline offline."""
    from expgym.envs.audit import AuditEnv
    from expgym.envs.search import SearchEnv
    from expgym.envs.tuning import TuningEnv

    rng = random.Random(seed)
    if isinstance(env, TuningEnv):
        configs = [{p.name: p.sample(rng) for p in env.params} for _ in range(n_actions)]
        return ScriptedBackend([("evaluate_config", c) for c in configs], json.dumps(configs[0]))
    if isinstance(env, SearchEnv):
        return ScriptedBackend([("search", {"query": env.question})], "")
    if isinstance(env, AuditEnv):
        actions = [("human_feedback", {"nda_id": h, "evidence_ids": []}) for h in env.order[:n_actions]]
        answer = {h: {"label": "NotMentioned", "evidence_ids": []} for h in env.order}
        return ScriptedBackend(actions, json.dumps(answer))
    raise ValueError("no scripted plan for %s" % type(env).__name__)


# ----------------------------------------------------------------------------
# Job selection and output
# ----------------------------------------------------------------------------

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True,
                        help="One of: %s, or a task registered by --import." % ", ".join(available_tasks()))
    parser.add_argument("--import", dest="imports", action="append", default=[], metavar="MODULE",
                        help="Python module or .py file to import first (e.g. to register a custom task).")
    parser.add_argument("--regimes", default="free,moderate,tight",
                        help="Comma-separated: free, moderate, tight or beta=<x> (default: all three).")
    parser.add_argument("--items", default="all",
                        help="'all', a group (e.g. whois, nasbench101), a slice such as 0:5, or comma-separated IDs.")
    parser.add_argument("--repeats", type=int,
                        help="Runs per item (default for run: 3 for tuning and audit, else 1; "
                             "for pool: 3 for tuning, else 1).")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-context-tokens", type=int, help="Optional rough client-side context cap.")
    parser.add_argument("--workers", type=int, default=4, help="Jobs run concurrently.")
    parser.add_argument("--output", required=True, help="Output directory; existing results are skipped.")
    add_backend_arguments(parser)


def select_items(task: Task, spec: str) -> List[str]:
    """Resolve ``all``, a slice such as ``0:5``, or a comma-separated list of
    item IDs and group names (e.g. ``whois`` or ``paramnet,nasbench101``)."""
    items = task.items()
    if spec == "all":
        return items
    if re.fullmatch(r"-?\d*:-?\d*", spec):
        start, stop = (int(x) if x else None for x in spec.split(":"))
        return items[start:stop]
    groups = [task.item_group(item) for item in items]
    chosen: List[str] = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        if part in items:
            chosen.append(part)
        elif part in groups:
            chosen += [item for item, group in zip(items, groups) if group == part]
        else:
            raise SystemExit("unknown item or group %r for task %s" % (part, task.name))
    return list(dict.fromkeys(chosen))


def import_plugins(paths: List[str]) -> None:
    for path in paths:
        if path.endswith(".py"):
            name = os.path.splitext(os.path.basename(path))[0]
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            importlib.import_module(path)


def load_task(name: str) -> Task:
    try:
        return get_task(name)
    except KeyError as exc:
        raise SystemExit(exc.args[0])


def parse_regimes(spec: str) -> List[str]:
    regimes = [r.strip() for r in spec.split(",") if r.strip()]
    for regime in regimes:
        try:
            get_regime(regime)
        except ValueError as exc:
            raise SystemExit(str(exc))
    return regimes


def model_name(args: argparse.Namespace) -> str:
    return args.model if args.backend == "openai" else args.backend


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)


def _write(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=1)
    os.replace(temporary, path)


def _run_jobs(jobs: List[Dict[str, Any]], workers: int, execute: Callable[[Dict[str, Any]], Dict[str, Any]]) -> int:
    pending = [job for job in jobs if not os.path.exists(job["path"])]
    print("%d jobs, %d already done, %d to run" % (len(jobs), len(jobs) - len(pending), len(pending)))
    failures = []
    print_lock = threading.Lock()

    def work(job: Dict[str, Any]) -> None:
        try:
            record = execute(job)
        except Exception as exc:  # keep the sweep going; report at the end
            with print_lock:
                print("FAILED %s: %s" % (job["path"], exc), file=sys.stderr)
            failures.append(job["path"])
            return
        _write(job["path"], record)
        with print_lock:
            print("done %s  %s" % (job["path"], json.dumps(record.get("summary", {}))))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, pending))
    if failures:
        print("%d job(s) failed; re-run the same command to retry them." % len(failures), file=sys.stderr)
    return 1 if failures else 0


def command_run(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    factory = make_backend_factory(args)
    repeats = args.repeats or DEFAULT_REPEATS.get(task.name, 1)
    jobs = []
    for regime in parse_regimes(args.regimes):
        for item in select_items(task, args.items):
            for repeat in range(repeats):
                path = os.path.join(args.output, _slug(model_name(args)), task.name, _slug(regime),
                                    "%s__r%d.json" % (_slug(item), repeat))
                jobs.append({"regime": regime, "item": item, "repeat": repeat, "path": path})

    def execute(job):
        env = task.make_env(job["item"], job["repeat"])
        result = run_agent(factory(env, job["repeat"]), env, get_regime(job["regime"]),
                           max_steps=args.max_steps, max_context_tokens=args.max_context_tokens)
        result["run"] = {"kind": "single", "task": task.name, "item": job["item"],
                         "group": task.item_group(job["item"]), "repeat": job["repeat"],
                         "regime": job["regime"], "model": model_name(args)}
        result["summary"] = result["score"]
        return result

    return _run_jobs(jobs, args.workers, execute)


def command_pool(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    factory = make_backend_factory(args)
    repeats = args.repeats or POOL_REPEATS.get(task.name, 1)
    strategies = args.strategies.split(",")
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise SystemExit("unknown strategy %r" % strategy)
    jobs = []
    for regime in parse_regimes(args.regimes):
        for strategy in strategies:
            for item in select_items(task, args.items):
                for repeat in range(repeats):
                    name = "%s__r%d.json" % (_slug(item), repeat)
                    label = strategy if args.lock or strategy != "poolact" else "poolact-nolock"
                    path = os.path.join(args.output, _slug(model_name(args)), task.name, _slug(regime),
                                        "%s_n%d" % (label, args.agents), name)
                    jobs.append({"regime": regime, "strategy": strategy, "item": item,
                                 "repeat": repeat, "path": path, "label": label})

    def execute(job):
        def make_backend(env: Environment, index: int) -> ChatBackend:
            return factory(env, job["repeat"] * args.agents + index)

        result = run_pool(task, job["item"], make_backend, job["strategy"], get_regime(job["regime"]),
                          n_agents=args.agents, repeat=job["repeat"], use_lock=args.lock,
                          max_steps=args.max_steps, max_context_tokens=args.max_context_tokens)
        result["run"] = {"kind": "pool", "task": task.name, "item": job["item"],
                         "group": task.item_group(job["item"]), "repeat": job["repeat"],
                         "regime": job["regime"], "strategy": job["label"], "agents": args.agents,
                         "model": model_name(args)}
        result["summary"] = {k: v for k, v in result["aggregate"].items() if k not in ("answer", "method")}
        return result

    return _run_jobs(jobs, args.workers, execute)


def command_summarize(args: argparse.Namespace) -> int:
    print(summarize(args.results, fmt=args.format))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="expgym", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Evaluate single agents under feedback budgets.")
    add_common_arguments(run)
    run.set_defaults(handler=command_run)

    pool = commands.add_parser("pool", help="Evaluate N-agent pools (naive / cached / poolact).")
    add_common_arguments(pool)
    pool.add_argument("--strategies", default="naive,cached,poolact")
    pool.add_argument("--agents", type=int, default=4)
    pool.add_argument("--no-lock", dest="lock", action="store_false",
                      help="Ablation: PoolAct without the decision lock.")
    pool.set_defaults(handler=command_pool, workers=1)

    report = commands.add_parser("summarize", help="Aggregate result files into tables.")
    report.add_argument("results", help="Output directory of `expgym run` / `expgym pool`.")
    report.add_argument("--format", choices=("markdown", "json"), default="markdown")
    report.add_argument("--import", dest="imports", action="append", default=[], metavar="MODULE",
                        help="Module registering custom tasks found in the results.")
    report.set_defaults(handler=command_summarize)

    args = parser.parse_args(argv)
    import_plugins(getattr(args, "imports", []))
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
