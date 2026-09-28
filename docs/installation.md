# Installation and benchmark data

[← README](../README.md)

## Installation

```bash
git clone https://github.com/tiannuo-yang/expgym.git
cd expgym
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev]"
```

The core supports Python 3.7+. Use a current Python for search, audit, NAS-Bench,
and development; ParamNet needs the separate environment below. On Windows,
activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.

## Benchmark data

Download the benchmark data. Everything goes under `data/` in the repository, or under `$EXPGYM_DATA_ROOT` if it is set.

```bash
python scripts/download_data.py --only search,audit      # ~70 MB download, ~5 MB kept
python scripts/download_data.py --only nasbench101,nasbench201
python scripts/download_data.py --only paramnet          # ~200 MB, needs git
python scripts/download_data.py --check                  # show what is present
```

The NAS-Bench downloads are large (2 GB and 1.1 GB). They are verified against pinned checksums, converted into small lookup tables (~25 MB and 1.4 MB), and then deleted. If you already have `nasbench_full.tfrecord`, `NATS-tss-v1_0-3ffb9-simple.tar` or the six ParamNet surrogate `.pkl` files on disk, point `NASBENCH101_TFRECORD`, `NATS_TSS_ARCHIVE` or `PARAMNET_SURROGATES_DIR` at them to skip the download.

## ParamNet environment

**ParamNet needs Python 3.7.** The three ParamNet tuning tasks use HPOBench surrogate models that were pickled with scikit-learn 0.23. From the repository root, leave the virtual environment if it is active, then use the [provided environment](../environment-paramnet.yml):

```bash
conda env create -f environment-paramnet.yml
conda activate expgym-paramnet
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/download_data.py --only paramnet
python scripts/download_data.py --only paramnet --check
```

The other tasks run on any Python ≥ 3.7.
