# ExpGym and PoolAct

Code for **"When Interaction Is the Bottleneck: Evaluating and Scaling Agents Under Costly Feedback"**.

LLM agents gather evidence by acting: they run experiments, retrieve documents and ask for verification. When that feedback is expensive, what an agent chooses to *ask* matters as much as how it reasons.

- **ExpGym** evaluates tool-using agents under explicit feedback budgets. Tool results come from pre-cached observations and deterministic surrogates, so you can re-run evaluations cheaply. Each request is still charged its simulated cost.
- **PoolAct** runs several agents on one task. They share completed observations and a live exploration graph, and a lock serializes action selection so agents avoid duplicating each other's requests.

The core package needs only NumPy (plus pyarrow for the search task). It works with any OpenAI-compatible chat endpoint (OpenAI, OpenRouter, vLLM, SGLang, ...), and you can add your own environments with a few dozen lines of code.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [How the benchmark works](#how-the-benchmark-works)
- [Running experiments](#running-experiments)
- [Reproducing the paper's settings](#reproducing-the-papers-settings)
- [Adding your own environment](#adding-your-own-environment)
- [Python API](#python-api)
- [Output format](#output-format)
- [Repository layout](#repository-layout)

## Installation

```bash
git clone <this repository> expgym && cd expgym
pip install -e ".[data,dev]"          # Python >= 3.7
```

Download the benchmark data. Everything goes under `data/` in the repository, or under `$EXPGYM_DATA_ROOT` if it is set.

```bash
python scripts/download_data.py --only search,audit      # ~70 MB download, ~5 MB kept
python scripts/download_data.py --only nasbench101,nasbench201
python scripts/download_data.py --only paramnet          # ~200 MB, needs git
python scripts/download_data.py --check                  # show what is present
```

The NAS-Bench downloads are large (2 GB and 1.1 GB). They are verified against pinned checksums, converted into small lookup tables (~25 MB and 1.4 MB), and then deleted. If you already have `nasbench_full.tfrecord`, `NATS-tss-v1_0-3ffb9-simple.tar` or the six ParamNet surrogate `.pkl` files on disk, point `NASBENCH101_TFRECORD`, `NATS_TSS_ARCHIVE` or `PARAMNET_SURROGATES_DIR` at them to skip the download.

**ParamNet needs Python 3.7.** The three ParamNet tuning tasks use HPOBench surrogate models that were pickled with scikit-learn 0.23. Run those tasks in the provided environment:

```bash
conda env create -f environment-paramnet.yml && conda activate expgym-paramnet
```

The other tasks run on any Python ≥ 3.7.

## Quick start

Check the pipeline offline with the scripted stand-in model. It needs no API key and no data:

```bash
python -m expgym run --task toy --backend scripted --output runs/smoke
python -m expgym summarize runs/smoke
```

Then run a real model. `--base-url` accepts any OpenAI-compatible endpoint, and the API key is read from `$EXPGYM_API_KEY` (use `--api-key-env` to read a different variable):

```bash
export EXPGYM_API_KEY=...
python -m expgym run --task search --items 0:5 --regimes tight \
    --model your-model --base-url https://your-endpoint/v1 --output runs/demo

python -m expgym pool --task audit --items 0 --regimes tight --agents 4 \
    --model your-model --base-url https://your-endpoint/v1 --output runs/demo

python -m expgym summarize runs/demo
```

After `pip install -e .`, the `expgym` command is equivalent to `python -m expgym`.

## How the benchmark works

### Environments

| Task (`--task`) | Items | Tool | Feedback cost | Metric |
|---|---|---|---|---|
| `tuning` | 9 HPOBench tasks: `paramnet/{adult,higgs,letter}`, `nasbench101/{A,B,C}`, `nasbench201/{cifar10-valid,cifar100,imagenet16-120}` | `evaluate_config` returns validation performance | the configuration's simulated training time | Gap Score |
| `search` | 73 three-hop PhantomWiki questions (39 *whois*, 34 *whatis*), IDs `seed2/0` … `seed3/36` | `search` returns one full article (exact title match, else keyword match) | 280–320 s for each *new* article; repeat retrievals are free | set F1 over names |
| `audit` | 13 ContractNLI documents × 17 hypotheses | `human_feedback` checks a proposed evidence set for one hypothesis | 280–320 s for every call | label accuracy, exact evidence-set accuracy |
| `toy` | a synthetic tuning problem (no data needed) | `evaluate_config` | synthetic | Gap Score |

The **Gap Score** normalizes tuning performance against a 100,000-sample random search of the same space. The statistics are shipped in `expgym/resources/hpo_reference.json`:

```
gap = max(0, (perf − mean_perf) / (best_perf − mean_perf) × 100)
```

- 0 means no better than the average random configuration, and 100 matches the best of 100K random samples. Scores above 100 are possible.
- The submitted configuration is scored from the matching evaluation the agent saw. If there is no match, the best evaluation the agent saw is used instead.
- If the agent saw no evaluation, the submitted configuration is evaluated offline.
- A run with no final answer, or with no scoreable configuration, gets 0.

### Feedback regimes

Every tool call is charged a simulated cost (seconds of simulated time, not wall-clock time or API cost). A regime caps the cumulative cost at `B = β · c_base`, where `c_base` is a task-specific reference cost:

- **tuning:** the cost of the best configuration in the reference search.
- **search and audit:** 300 s.

| `--regimes` | β | Cost shown to the agent |
|---|---|---|
| `free` | ∞ | no |
| `moderate` | 10 | yes: `Observation: … \| cost=293s [time_left=2415s]` |
| `tight` | 3 | yes |
| `beta=<x>` | x | yes (used for budget sweeps) |

The agent is a ReAct loop with native function calling. Each turn may call one tool or give a final answer. Budget and step limits work as follows:

- **Budget boundary:** the call that brings the spent cost to `B` or beyond is charged, but its result is withheld.
- **Step limit:** every run has a limit of 30 steps, and zero-cost calls count toward it.
- **Forced answer:** if either limit is hit, the agent is asked to answer immediately from what it has already seen, with tools disabled.

### PoolAct and the baselines

`expgym pool` runs `N` agents (default 4) on the same item. Every agent has the full per-agent budget and step limit. The strategies differ only in what the agents share:

| `--strategies` | Shared completed results | Shared exploration graph | Decision lock |
|---|---|---|---|
| `naive` | – | – | – |
| `cached` | ✓ | – | – |
| `poolact` | ✓ | ✓ | ✓ |
| `poolact --no-lock` (ablation) | ✓ | ✓ | – |

Agents run on simulated clocks: an agent's clock equals its own spent cost. How agents see each other's work:

- **Cached results:** a request matches a completed one when the tool name and the arguments (as JSON with sorted keys) are equal. Another agent's result becomes visible, and free to reuse, only once the requesting agent's clock has passed that result's completion time.
- **The exploration graph:** appended to the newest observation before every PoolAct decision. It lists requests in progress, results already explored, per-agent exploration paths and a coverage summary.
- **The decision lock:** covers reading the graph, the model call and registering the chosen request. It is released before the tool runs.
- **Withheld results:** results withheld at a budget boundary are never shared, and the request is no longer shown as in progress.

How pool results are scored:

- **Tuning:** the mean Gap Score of the agents (the best-of-N score is reported too).
- **Search:** majority vote over normalized answer sets.
- **Audit:** a per-hypothesis vote, first on the label and then on the evidence set among agents that chose the winning label. The pool is scored by exact evidence-set accuracy; label accuracy is also saved in each result file.

## Running experiments

```bash
python -m expgym run  --task TASK [--regimes free,moderate,tight] [--items ...] [--repeats R] --model M --output DIR
python -m expgym pool --task TASK [--strategies naive,cached,poolact] [--agents 4] ... --output DIR
python -m expgym summarize DIR [--format json]
```

- `--items` accepts `all` (default), a group (`whois`, `whatis`, `paramnet`, `nasbench101`, `nasbench201`), a slice such as `0:10`, or a comma-separated list of item IDs.
- `--items` also accepts several groups at once, e.g. `nasbench101,nasbench201`.
- `--repeats` defaults to 3 for `tuning` and `audit` and to 1 otherwise. Audit repeats use three fixed hypothesis orderings. For `pool`, tuning uses 3 repeats and the other tasks use 1.
- `--workers` controls how many jobs run concurrently (4 for `run`, 1 for `pool`, whose agents already run in parallel).
- Every job writes one JSON file under `DIR/<model>/...`, and re-running the same command skips finished jobs. If a job fails (for example because of an API error), the others continue, and the next invocation retries it. Use a new output directory when you change settings other than the model.
- `--import MODULE_OR_FILE.py` imports a module first, e.g. one that registers a custom task (see below).

Model options:

| Flag | Meaning |
|---|---|
| `--model`, `--base-url`, `--api-key-env` | OpenAI-compatible endpoint (`$EXPGYM_MODEL` and `$EXPGYM_BASE_URL` are used as defaults) |
| `--temperature`, `--top-p`, `--max-tokens`, `--seed` | sampling; unset values use the server's defaults. Each run and each pool agent gets its own seed offset from `--seed` |
| `--extra-body JSON` | extra request fields, e.g. `'{"reasoning_effort": "high"}'`, `'{"top_k": 20}'`, `'{"chat_template_kwargs": {"enable_thinking": true}}'` |
| `--max-context-tokens N` | client-side context cap; older tool outputs are shortened, then dropped |
| `--max-steps N` | interaction limit (default 30) |

`summarize` averages repeats within an item, then items within a group, then groups:

- **Tuning:** the three benchmark families get equal weight.
- **Search:** *whois* and *whatis* get equal weight.
- **Audit:** label accuracy and exact evidence-set accuracy are reported separately.

A task's AVG is the mean of its headline metrics, and **Overall AVG** is the mean of the tuning, search and audit AVGs (shown when all three are present). Custom tasks are summarized by their primary metric.

## Reproducing the paper's settings

Single-agent evaluation (main results table), per model:

```bash
M=your-model
python -m expgym run --task search --model $M --output runs
python -m expgym run --task audit  --model $M --output runs
python -m expgym run --task tuning --items nasbench101,nasbench201 --model $M --output runs
python -m expgym run --task tuning --items paramnet --model $M --output runs   # in the Python 3.7 env
python -m expgym summarize runs
```

That is 9 tuning tasks × 3 repeats, 73 questions and 13 documents × 3 orderings, each under the three regimes. The paper's generation settings for each model (output cap, sampling, reasoning effort) are listed in its appendix; pass them with `--max-tokens`, `--temperature`, `--top-p` and `--extra-body`.

4-agent pools (PoolAct results table):

```bash
python -m expgym pool --task search --items whois --regimes moderate,tight --max-context-tokens 131072 --model $M --output pools
python -m expgym pool --task audit  --regimes moderate,tight --max-context-tokens 131072 --model $M --output pools
python -m expgym pool --task tuning --items nasbench101 --regimes moderate,tight --max-context-tokens 131072 --model $M --output pools
python -m expgym summarize pools
```

Variations:

- **Pool size:** `--agents 2`, `6` or `8` (results go to `<strategy>_n<N>` directories and appear as separate rows).
- **Lock ablation:** `--strategies poolact --no-lock` (results go to `poolact-nolock_n<N>`).
- **Budget sweep:** `--task search --items whois --regimes beta=1,beta=5,beta=10,beta=15,beta=20`.

LLM sampling is stochastic, so expect run-to-run variance. The paper reports its variability across repeats in the appendix.

## Adding your own environment

An environment is one task instance. It exposes function-calling tools, charges a simulated cost for each call and scores the final answer:

```python
from expgym import register_task, run_agent, get_regime
from expgym.envs.base import Environment, Task, ToolResult, ToolInputError, function_tool

class MyEnv(Environment):
    reference_cost = 3600.0                     # c_base: budget is beta * c_base

    def task_prompt(self):                      # first user message
        return "..."

    def tools(self):                            # OpenAI function definitions
        return [function_tool("measure", "Run one measurement.", {...json schema...})]

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

```python
from expgym import get_task, get_regime, run_agent, run_pool
from expgym.llm import OpenAIChatBackend

task = get_task("search")
env = task.make_env("seed2/0")
backend = OpenAIChatBackend("your-model", "https://your-endpoint/v1", api_key="...")
result = run_agent(backend, env, get_regime("tight"))
print(result["score"], result["feedback_cost"], result["termination"])

pool = run_pool(task, "seed2/0", lambda env, i: OpenAIChatBackend(...), "poolact",
                get_regime("tight"), n_agents=4)
print(pool["aggregate"])
```

Any object with a `generate(messages, tools, tool_choice)` method that returns an `expgym.llm.ModelTurn` can serve as the backend.

## Output format

`expgym run` writes one file per trajectory, at `DIR/<model>/<task>/<regime>/<item>__r<repeat>.json`:

| Field | Content |
|---|---|
| `run` | `kind` (`single` or `pool`), task, item, group, repeat, regime, model (pools also record `strategy` and `agents`) |
| `answer`, `score` | the final answer and its metrics (tuning also reports `perf` and which rule scored it) |
| `termination` | `answered`, `budget_exhausted`, `step_limit` or `invalid_response` |
| `feedback_cost`, `budget` | simulated cost spent, and the budget `B` (`null` under `free`) |
| `steps` | model turns, including the forced final answer |
| `tool_calls` | every call, with its arguments, cost, `perf`, `reused` (shared-cache hit) and `visible` (whether the result was shown) |
| `usage` | model calls, prompt/completion tokens and model latency |
| `messages` | the complete conversation |
| `summary` | the metrics printed while the sweep runs |

`expgym pool` writes `DIR/<model>/<task>/<regime>/<strategy>_n<N>/<item>__r<repeat>.json`. Each file contains `aggregate` (the pool's score), the shared-cache statistics and the full record of every agent under `agents`.

## Repository layout

```
expgym/
  agent.py         ReAct loop, feedback regimes and budget accounting
  poolact.py       naive / cached / PoolAct strategies, shared cache and exploration graph
  llm.py           OpenAI-compatible backend and the scripted test backend
  envs/            tuning (HPOBench + NAS-Bench tables), search (PhantomWiki), audit (ContractNLI), base classes
  summary.py       aggregation used by `expgym summarize`
  cli.py           command-line interface
  resources/       random-search reference statistics, audit hypothesis orders, evidence hint labels
scripts/download_data.py   fetch, verify and convert benchmark data
examples/                  a custom environment
tests/                     unit and end-to-end tests (`pytest`)
```

## Citation

```bibtex
@inproceedings{expgym2026,
  title  = {When Interaction Is the Bottleneck: Evaluating and Scaling Agents Under Costly Feedback},
  author = {Anonymous Authors},
  year   = {2026}
}
```

## License

The code is released under the MIT License. The benchmark data keep their original licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
