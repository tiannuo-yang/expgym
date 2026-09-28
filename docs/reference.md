# Output format and repository layout

[← README](../README.md)

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
