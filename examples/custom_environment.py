"""Plug your own costly-feedback task into ExpGym.

This toy "wet lab" asks the agent to find the most active compound in a
library. Each assay costs simulated hours, so the agent has to decide which
compounds are worth testing under the feedback budget.

Run offline with a scripted stand-in for the model:

    python examples/custom_environment.py

or with a real model through any OpenAI-compatible endpoint:

    EXPGYM_API_KEY=... python examples/custom_environment.py --model your-model

The task can also be run with the command-line runner:

    python -m expgym run --import examples/custom_environment.py --task assay \
        --regimes free,tight --model your-model --output runs/assay
    python -m expgym summarize runs/assay --import examples/custom_environment.py
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expgym import get_regime, register_task, run_agent, run_pool  # noqa: E402
from expgym.envs.base import Environment, Task, ToolInputError, ToolResult, function_tool, stable_uniform  # noqa: E402
from expgym.llm import OpenAIChatBackend, ScriptedBackend  # noqa: E402


class AssayEnv(Environment):
    """One screening campaign over ``n`` compounds with hidden activities."""

    reference_cost = 3600.0  # c_base: one assay takes about an hour, so B = beta hours

    def __init__(self, seed: int, n: int = 40) -> None:
        rng = random.Random(seed)
        self.activity = {"C%02d" % i: round(rng.betavariate(2, 5), 3) for i in range(n)}
        self.seed = seed

    def task_prompt(self) -> str:
        return ("A library contains compounds %s. Each has an unknown activity in [0, 1]. "
                "Use the assay tool to measure compounds, then answer with the ID of the most "
                "active compound, e.g. Answer: C07." % ", ".join(sorted(self.activity)))

    def tools(self):
        return [function_tool("assay", "Measure the activity of one compound (takes about an hour).",
                              {"type": "object", "properties": {"compound": {"type": "string"}},
                               "required": ["compound"], "additionalProperties": False})]

    def call(self, name, arguments):
        compound = str(arguments.get("compound", "")).strip().upper()
        if compound not in self.activity:
            raise ToolInputError("unknown compound %r" % compound)  # shown to the model, no charge
        cost = stable_uniform("assay:%d:%s" % (self.seed, compound), 3000.0, 4200.0)
        return ToolResult("activity=%.3f" % self.activity[compound], cost, self.activity[compound])

    def score(self, answer, observations):
        chosen = (answer or "").strip().upper()
        best = max(self.activity.values())
        return {"regret": best - self.activity.get(chosen, 0.0), "found_best": float(self.activity.get(chosen) == best)}

    # Optional: how actions appear in PoolAct's shared exploration graph.
    def describe_action(self, name, arguments):
        return "assay %s" % arguments.get("compound")

    def describe_outcome(self, name, arguments, result):
        return "assay %s -> %s" % (arguments.get("compound"), result.text)


class AssayTask(Task):
    name = "assay"
    pool_aggregation = "vote"

    def items(self):
        return [str(seed) for seed in range(10)]

    def make_env(self, item, repeat=0):
        return AssayEnv(int(item))

    def primary_metric(self):
        return "found_best"

    def vote(self, answers):
        answers = [a.strip().upper() for a in answers if a]
        return max(set(answers), key=answers.count) if answers else None


register_task("assay", AssayTask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Use a real model instead of the scripted stand-in.")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    args = parser.parse_args()

    def backend(env, index=0):
        if args.model:
            return OpenAIChatBackend(args.model, args.base_url, os.environ.get("EXPGYM_API_KEY"))
        picks = random.Random(index).sample(sorted(env.activity), 6)
        return ScriptedBackend([("assay", {"compound": c}) for c in picks], picks[0])

    task = AssayTask()
    for regime in ("free", "beta=3"):
        env = task.make_env("0")
        result = run_agent(backend(env), env, get_regime(regime))
        print("%-7s single agent: %s, answer=%s, cost=%.0fs, termination=%s"
              % (regime, result["score"], result["answer"], result["feedback_cost"], result["termination"]))
    pool = run_pool(task, "0", backend, "poolact", get_regime("beta=3"), n_agents=3)
    print("beta=3  PoolAct x3:  ", pool["aggregate"])


if __name__ == "__main__":
    main()
