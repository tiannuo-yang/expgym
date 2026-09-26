# Third-party data and code

ExpGym does not redistribute benchmark datasets. `scripts/download_data.py`
fetches them from their original sources; each remains under its own license.

| Component | Used for | Source | License |
|---|---|---|---|
| PhantomWiki v1 (Gong et al., 2025) | Multi-hop search corpus and questions | [kilian-group/phantom-wiki-v1](https://huggingface.co/datasets/kilian-group/phantom-wiki-v1), revision `9369f9c6` | MIT |
| ContractNLI (Koreeda & Manning, 2021) | Evidence audit documents and annotations | [stanfordnlp.github.io/contract-nli](https://stanfordnlp.github.io/contract-nli/) | CC BY 4.0 |
| HPOBench (Eggensperger et al., 2021) | ParamNet surrogates and task definitions | [automl/HPOBench](https://github.com/automl/HPOBench), commit `47bf141f` | Apache-2.0 |
| NAS-Bench-101 (Ying et al., 2019) | NAS-Bench-101 tabular results | [google-research/nasbench](https://github.com/google-research/nasbench) | Apache-2.0 |
| NAS-Bench-201 / NATS-Bench (Dong et al.) | NAS-Bench-201 tabular results (topology search space) | [D-X-Y/NATS-Bench](https://github.com/D-X-Y/NATS-Bench) | MIT |
| ParamNet surrogate models (HPOlib) | ParamNet tuning tasks | [LoneKnightz/HPOlib2](https://github.com/LoneKnightz/HPOlib2) (mirror of the HPOlib `surrogates` branch), commit `de88ab3a` | GPL-3.0 (repository) |

Files shipped in `expgym/resources/` are derived from these benchmarks:

- `hpo_reference.json`: statistics of a 100,000-sample random search on each
  tuning task (mean and best validation performance, cost and configuration of
  the best sample).
- `contract_nli_hints.json`: a coarse, human-readable category for each
  evidence segment of the audit documents, used in `Evidence Incomplete` feedback.
- `audit_hypothesis_orders.json`: three fixed random orderings of the 17
  ContractNLI hypotheses.

The ParamNet surrogate files are downloaded from a pinned commit of a public
HPOlib mirror and verified against fixed git blob hashes. If that mirror becomes
unavailable, place the six files in a directory and set `PARAMNET_SURROGATES_DIR`.
