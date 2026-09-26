import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expgym.envs.base import Environment, Task, ToolInputError, ToolResult, function_tool  # noqa: E402


class CounterEnv(Environment):
    """Every ``probe`` call costs ``cost`` seconds and returns its argument doubled."""

    reference_cost = 100.0

    def __init__(self, cost=100.0):
        self.cost = cost

    def task_prompt(self):
        return "Probe numbers, then answer with the largest result."

    def tools(self):
        return [function_tool("probe", "Double a number.",
                              {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]})]

    def call(self, name, arguments):
        if not isinstance(arguments.get("x"), int):
            raise ToolInputError("x must be an integer")
        return ToolResult("value=%d" % (2 * arguments["x"]), self.cost, float(2 * arguments["x"]))

    def score(self, answer, observations):
        return {"correct": float(answer == "42"), "seen": len(observations)}


class CounterTask(Task):
    name = "counter"
    pool_aggregation = "vote"

    def items(self):
        return ["0"]

    def make_env(self, item, repeat=0):
        return CounterEnv()

    def primary_metric(self):
        return "correct"

    def vote(self, answers):
        return max(set(answers), key=answers.count)


@pytest.fixture
def counter_env():
    return CounterEnv()
