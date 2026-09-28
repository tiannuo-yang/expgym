# Benchmark and PoolAct protocol

[← README](../README.md)

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
