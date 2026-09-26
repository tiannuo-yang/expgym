"""Data locations. Everything lives under ``$EXPGYM_DATA_ROOT`` (default ``./data``)."""
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHANTOM_WIKI_REPO = "kilian-group/phantom-wiki-v1"
PHANTOM_WIKI_REVISION = "9369f9c64655f4e8146afee75ae5d3e3a95d7df5"


def data_root() -> str:
    return os.environ.get("EXPGYM_DATA_ROOT", os.path.join(REPO_ROOT, "data"))


def phantom_wiki_dir() -> str:
    """Directory containing ``question-answer/`` and ``text-corpus/``."""
    return os.path.join(data_root(), "phantom-wiki")


def contract_nli_file() -> str:
    return os.path.join(data_root(), "contract-nli", "test.json")


def hpo_dir() -> str:
    """HPOBench checkout, ParamNet surrogates and compact NAS tables."""
    return os.path.join(data_root(), "hpo")


def resource(*parts: str) -> str:
    """Small files shipped with the package (reference scores, hints, orders)."""
    return os.path.join(REPO_ROOT, "expgym", "resources", *parts)
