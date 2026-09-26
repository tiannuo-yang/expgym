"""Hyperparameter tuning on deterministic HPOBench surrogates and NAS tables.

The agent calls ``evaluate_config`` with a complete configuration and receives
its validation performance; each evaluation is charged the benchmark's
simulated training cost. The final configuration is scored with the Gap Score
against a 100K-sample random search of the same space::

    gap = max(0, (perf - mean_perf) / (best_perf - mean_perf) * 100)

Scoring rule for the submitted answer: if it matches a configuration the
agent evaluated (and saw), that result is used; otherwise the best visible
evaluation is used; if the agent saw no evaluation, the submitted configuration
is evaluated offline. Runs without a scoreable configuration receive 0.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from expgym.envs.base import Environment, Task, ToolInputError, ToolResult, canonical_json, function_tool
from expgym.envs.nas import NasBench101, NasBench201, nas101_edge_pairs
from expgym.envs.params import Param, params_from_configspace, validate_config
from expgym.paths import hpo_dir, resource

TASKS = (
    "paramnet/adult", "paramnet/higgs", "paramnet/letter",
    "nasbench101/A", "nasbench101/B", "nasbench101/C",
    "nasbench201/cifar10-valid", "nasbench201/cifar100", "nasbench201/imagenet16-120",
)

NAS201_HINT = (
    "Note: This is a neural architecture search task. "
    "Each parameter selects an operation for an edge in a fixed cell structure. "
    "All operations are valid; explore diverse combinations for better performance."
)

NAS101_HINT = (
    "Note: This is a neural architecture search task on a "
    "7-node directed acyclic graph (DAG). Node 0 is the input and node 6 "
    "is the output; op_node_0 through op_node_4 specify operations on "
    "nodes 1 through 5. The decoded graph must have AT MOST 9 edges "
    "and a connected path from node 0 to node 6. Graphs with more than "
    "9 edges or no input-to-output path will always score 0. "
    "You must specify ALL parameters in every evaluation call. "
)

NAS101_ENCODING = {
    "A": ("Each of the 21 edge_0 through edge_20 parameters is a binary "
          "adjacency entry (0=absent, 1=present), ordered row by row in "
          "the upper-triangular adjacency matrix. Set unused edges to 0; "
          "do not set all 21 entries to 1. Edge parameter mapping: "),
    "B": ("The 9 parameters edge_0 through edge_8 are edge-ID selectors, "
          "each an integer from 0 to 20, NOT binary adjacency entries. "
          "Each selector chooses one edge from the mapping below. "
          "Repeated IDs select that edge only once, so duplicates can "
          "represent fewer than 9 edges. There is no unused sentinel: "
          "ID 0 is a real edge (5->6), not an absent edge. IDs use the "
          "decoder's reversed column-wise bit order. Edge-ID mapping: "),
    "C": ("The 21 parameters edge_0 through edge_20 are real-valued "
          "edge priorities in [0,1], NOT binary adjacency entries. "
          "num_edges is an integer from 0 to 9; exactly the num_edges "
          "highest-priority edges are selected (top-k). A priority of "
          "0 is not an absent-edge sentinel; selection depends on rank. "
          "Use distinct priorities to avoid tie ambiguity. num_edges=0 "
          "selects no edges and cannot connect input to output. Priorities "
          "use row-wise upper-triangular order. Edge parameter mapping: "),
}


def task_hint(task: str) -> Optional[str]:
    family, _, variant = task.partition("/")
    if family == "nasbench201":
        return NAS201_HINT
    if family == "nasbench101":
        prefix = "" if variant == "B" else "edge_"
        mapping = ", ".join("%s%d=%d->%d" % (prefix, i, s, t)
                            for i, (s, t) in enumerate(nas101_edge_pairs(variant)))
        return NAS101_HINT + NAS101_ENCODING[variant] + mapping + "."
    return None


class ParamNet:
    """HPOBench ParamNet surrogate.

    The surrogate forests were pickled with scikit-learn 0.23 and need Python 3.7;
    see ``environment-paramnet.yml``.
    """

    fidelity = {"step": 50}

    def __init__(self, dataset: str) -> None:
        root = os.path.join(hpo_dir(), "HPOBench")
        surrogate = os.path.join(hpo_dir(), "hpobench_data", "Surrogates", "rf_surrogate_paramnet_%s.pkl" % dataset)
        for path in (root, surrogate):
            if not os.path.exists(path):
                raise FileNotFoundError("%s not found; run `python scripts/download_data.py --only paramnet`" % path)
        if root not in sys.path:
            sys.path.insert(0, root)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r"Trying to unpickle estimator")
        import hpobench
        from hpobench.benchmarks.surrogates import paramnet_benchmark

        # Read the surrogates from our data root, not from HPOBench's global config.
        hpobench.config_file.data_dir = Path(hpo_dir(), "hpobench_data")
        hpobench.config_file.cache_dir = Path(hpo_dir(), "hpobench_cache")

        self.benchmark = getattr(paramnet_benchmark, "ParamNet%sOnStepsBenchmark" % dataset.capitalize())(rng=1)
        self._params = params_from_configspace(self.benchmark.get_configuration_space(seed=1))

    def params(self) -> List[Param]:
        return self._params

    def evaluate(self, config: Dict[str, Any]) -> Tuple[float, float]:
        result = self.benchmark.objective_function(configuration=config, fidelity=self.fidelity)
        return float(result["function_value"]), float(result["cost"])


def load_benchmark(task: str):
    family, _, variant = task.partition("/")
    if family == "paramnet":
        return ParamNet(variant)
    if family == "nasbench101":
        return NasBench101(variant)
    if family == "nasbench201":
        return NasBench201(variant)
    raise ValueError("unknown tuning task %r; choose from %s" % (task, ", ".join(TASKS)))


def error_to_perf(function_value: float) -> float:
    """Benchmarks report validation error as a fraction or a percentage."""
    perf = 1.0 - function_value / 100.0 if function_value > 1.0 else 1.0 - function_value
    return min(1.0, max(0.0, perf))


def parse_config(answer: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse the JSON configuration in an answer (optionally in a code fence)."""
    if not answer:
        return None
    text = answer.strip().rstrip(";").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class TuningEnv(Environment):
    def __init__(self, task: str, benchmark: Any, reference: Dict[str, float],
                 lock: Optional[threading.Lock] = None) -> None:
        self.task = task
        self.benchmark = benchmark
        # Alphabetical, matching the order in which ConfigSpace lists parameters.
        self.params = sorted(benchmark.params(), key=lambda p: p.name)
        self.reference = reference
        self.reference_cost = reference["best_cost"]
        self.lock = lock or threading.Lock()

    def task_prompt(self):
        lines = ["Goal: tune the system for high performance.", "Parameter ranges:"]
        lines += [p.describe() for p in self.params]
        hint = task_hint(self.task)
        if hint:
            lines.append(hint)
        lines += ['Call the evaluate_config function with arguments {"param": value, ...} using a native tool call.',
                  'Answer format: Answer: {"param": value, ...}']
        return "\n".join(lines)

    def tools(self):
        description = ("Evaluate a complete flat configuration for the described system. "
                       "Return performance and simulated evaluation cost; fidelity is fixed at %s "
                       "and is not an input field." % json.dumps(self.benchmark.fidelity, sort_keys=True))
        hint = task_hint(self.task)
        if hint:
            description += " " + hint
        properties = {p.name: p.schema() for p in self.params}
        return [function_tool("evaluate_config", description,
                              {"type": "object", "properties": properties,
                               "required": list(properties), "additionalProperties": False})]

    def _evaluate(self, config: Dict[str, Any]) -> Tuple[float, float]:
        with self.lock:  # surrogate models are shared between concurrent agents
            function_value, cost = self.benchmark.evaluate(config)
        return error_to_perf(function_value), cost

    def call(self, name, arguments):
        config = validate_config(self.params, arguments, ignored=list(self.benchmark.fidelity))
        perf, cost = self._evaluate(config)
        if perf == 0.0:
            return ToolResult("perf=0.000000 (invalid or degenerate configuration, try a different one)", cost, 0.0)
        return ToolResult("perf=%.6f" % perf, cost, perf)

    def gap(self, perf: float) -> float:
        mean, best = self.reference["mean_perf"], self.reference["best_perf"]
        return max(0.0, (perf - mean) / (best - mean) * 100.0)

    def score(self, answer, observations):
        evaluated = [o for o in observations if o.result.perf is not None]
        config = parse_config(answer)
        perf, source = None, "none"
        if answer is not None:
            key = canonical_json(config) if config is not None else None
            matches = [o for o in evaluated if canonical_json(o.arguments) == key]
            if matches:
                perf, source = matches[-1].result.perf, "matched_evaluation"
            elif evaluated:
                perf, source = max(o.result.perf for o in evaluated), "best_visible_evaluation"
            elif config is not None:
                try:
                    perf, _ = self._evaluate(validate_config(self.params, config, list(self.benchmark.fidelity)))
                    source = "offline_evaluation"
                except ToolInputError:
                    pass
        return {"gap": self.gap(perf) if perf is not None else 0.0, "perf": perf, "scored_from": source}

    def describe_action(self, name, arguments):
        return "evaluate {%s}" % ", ".join(
            "%s=%s" % (k, ("%.4g" % v) if isinstance(v, float) else v) for k, v in sorted(arguments.items()))

    def describe_outcome(self, name, arguments, result):
        config = "{" + ",".join("%s:%s" % (k, v) for k, v in sorted(arguments.items())) + "}"
        return "%s -> %s" % (config, "%.6f" % result.perf if result.perf else "INVALID")

    def coverage_note(self, outcomes):
        valid = [o.result.perf for o in outcomes if o.result.perf]
        if not valid:
            return "No valid configuration found yet."
        return "%d unique configs evaluated, best: %.6f" % (len(outcomes), max(valid))


class TuningTask(Task):
    """Items are benchmark names; ``repeat`` indexes independent runs."""

    name = "tuning"
    pool_aggregation = "mean"

    def __init__(self, tasks: Sequence[str] = TASKS) -> None:
        self.tasks = list(tasks)
        with open(resource("hpo_reference.json"), encoding="utf-8") as handle:
            self.references = json.load(handle)
        self._benchmarks: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def items(self):
        return list(self.tasks)

    def make_env(self, item, repeat=0):
        with self._guard:
            if item not in self._benchmarks:
                self._benchmarks[item] = load_benchmark(item)
                self._locks[item] = threading.Lock()
        return TuningEnv(item, self._benchmarks[item], self.references[item], self._locks[item])

    def item_group(self, item):
        return item.split("/")[0]

    def primary_metric(self):
        return "gap"

    def vote(self, answers):
        raise NotImplementedError("tuning pools report agent scores, not a voted answer")


# ----------------------------------------------------------------------------
# A dependency-free synthetic task for smoke tests and quick demos.
# ----------------------------------------------------------------------------

class ToyBenchmark:
    """Smooth synthetic training-performance surface with a cost model (no data needed).

    Its Gap Score reference uses 20,000 random samples, computed on first use.
    """

    fidelity: Dict[str, Any] = {}

    def params(self) -> List[Param]:
        return [Param.integer("num_layers", 2, 12), Param.integer("hidden_width", 64, 768),
                Param.real("learning_rate", 1e-5, 1e-2, log=True), Param.real("dropout", 0.0, 0.5),
                Param.integer("batch_size", 8, 512, log=True)]

    def evaluate(self, config: Dict[str, Any]) -> Tuple[float, float]:
        depth = (config["num_layers"] - 2) / 10.0
        width = (config["hidden_width"] - 64) / 704.0
        capacity = 0.25 * math.tanh(0.8 * depth + 0.9 * width)
        lr = math.exp(-((math.log10(config["learning_rate"]) + 3.5) ** 2) / (2 * 0.3 ** 2))
        drop = math.exp(-((config["dropout"] - 0.15) ** 2) / (2 * 0.08 ** 2))
        batch = math.exp(-((math.log2(config["batch_size"]) - 6) ** 2) / (2 * 1.2 ** 2))
        perf = 0.35 + capacity + 0.2 * lr + 0.1 * drop + 0.08 * batch - 0.2 * max(0.0, depth + width - 1.4)
        cost = 10.0 + 60.0 * (0.5 + depth) * (0.5 + width) * (256.0 / config["batch_size"]) ** 0.3
        return 1.0 - min(1.0, max(0.0, perf)), cost


class ToyTuningTask(Task):
    name = "toy"
    pool_aggregation = "mean"

    def __init__(self) -> None:
        self.benchmark = ToyBenchmark()
        rng = random.Random(0)
        params = self.benchmark.params()
        samples = [self.benchmark.evaluate({p.name: p.sample(rng) for p in params}) for _ in range(20000)]
        perfs = [error_to_perf(value) for value, _ in samples]
        best = max(range(len(perfs)), key=perfs.__getitem__)
        self.reference = {"mean_perf": sum(perfs) / len(perfs), "best_perf": perfs[best],
                          "best_cost": samples[best][1]}

    def items(self):
        return ["toy"]

    def make_env(self, item, repeat=0):
        return TuningEnv("toy", self.benchmark, self.reference)

    def primary_metric(self):
        return "gap"

    def vote(self, answers):
        raise NotImplementedError("tuning pools report agent scores, not a voted answer")
