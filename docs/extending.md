# Custom environments and Python API

[← README](../README.md)

## Adding your own environment

An environment is one task instance. It exposes function-calling tools, charges a simulated cost for each call and scores the final answer. The template below shows the interface; replace its placeholders before running it. For a complete runnable implementation, see [the compound-screening example](../examples/custom_environment.py).

```python
from expgym import register_task, run_agent, get_regime
from expgym.envs.base import Environment, Task, ToolResult, ToolInputError, function_tool

class MyEnv(Environment):
    reference_cost = 3600.0                     # c_base: budget is beta * c_base

    def task_prompt(self):                      # first user message
        return "..."

    def tools(self):                            # OpenAI function definitions
        schema = {"type": "object", "properties": {}}  # add your tool arguments
        return [function_tool("measure", "Run one measurement.", schema)]

    def call(self, name, arguments):            # raise ToolInputError for bad input (free)
        return ToolResult(text="value=0.42", cost=3500.0)

    def score(self, answer, observations):      # observations = results the agent saw
        return {"accuracy": ...}

class MyTask(Task):
    name = "mytask"
    def items(self): return ["0", "1", ...]
    def make_env(self, item, repeat=0): return MyEnv(...)
    def primary_metric(self): return "accuracy"
    def vote(self, answers): ...                # combine pool answers

register_task("mytask", MyTask)
```

Run it from the command line with `python -m expgym run --import path/to/my_task.py --task mytask ...` (pass the same `--import` to `summarize`).

`examples/custom_environment.py` is a complete, runnable example: a compound-screening task in which every assay costs about an hour of simulated time. It includes single-agent and PoolAct runs. Environments can also override `describe_action`, `describe_outcome` and `coverage_note` to control how their actions appear in PoolAct's exploration graph, and `on_reuse` to react when a result is reused from another agent.

## Python API

Download the search data and set `EXPGYM_MODEL`, `EXPGYM_BASE_URL`, and
`EXPGYM_API_KEY` as described in the [README](../README.md#use-your-model).
This example makes real model calls:

```python
import os

from expgym import get_task, get_regime, run_agent, run_pool
from expgym.llm import OpenAIChatBackend


def make_backend(env, index):
    return OpenAIChatBackend(
        os.environ["EXPGYM_MODEL"],
        os.environ["EXPGYM_BASE_URL"],
        api_key=os.environ.get("EXPGYM_API_KEY"),
    )


task = get_task("search")
env = task.make_env("seed2/0")
result = run_agent(make_backend(env, 0), env, get_regime("tight"))
print(result["score"], result["feedback_cost"], result["termination"])

pool = run_pool(task, "seed2/0", make_backend, "poolact",
                get_regime("tight"), n_agents=4)
print(pool["aggregate"])
```

Any object with a `generate(messages, tools, tool_choice)` method that returns an `expgym.llm.ModelTurn` can serve as the backend.
