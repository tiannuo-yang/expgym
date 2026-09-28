# Running experiments and reproducing settings

[← README](../README.md)

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
- `--import MODULE_OR_FILE.py` imports a module first, e.g. one that registers a [custom task](extending.md).

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

Prepare the [benchmark datasets and ParamNet environment](installation.md)
first. These commands specify the task matrix documented by the source
release. Exact reproduction also requires the paper's model versions and
generation settings; the CLI defaults alone do not establish paper fidelity.

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
