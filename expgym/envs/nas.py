"""Table-lookup NAS-Bench-101 / NAS-Bench-201 backends at full fidelity.

``scripts/download_data.py`` converts the official releases into small pickle
tables keyed by architecture (NAS-Bench-101 at 108 epochs, NAS-Bench-201 at
200 epochs), so these benchmarks need neither TensorFlow nor HPOBench at run
time. The search spaces and encodings follow HPOBench.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

from expgym.envs.params import Param
from expgym.paths import hpo_dir

NAS101_SCHEMA = "expgym.nasbench101"
NAS201_SCHEMA = "expgym.nasbench201"

# ----------------------------------------------------------------------------
# NAS-Bench-201
# ----------------------------------------------------------------------------

NAS201_OPERATIONS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
NAS201_EDGES = ("1<-0", "2<-0", "2<-1", "3<-0", "3<-1", "3<-2")
NAS201_DATASETS = ("cifar10-valid", "cifar100", "imagenet16-120")


def nas201_id(config: Dict[str, Any]) -> str:
    """Base-5 architecture ID used as the table key."""
    return "".join(str(NAS201_OPERATIONS.index(config[edge])) for edge in NAS201_EDGES)


class NasBench201:
    fidelity = {"epoch": 200}

    def __init__(self, dataset: str, path: Optional[str] = None) -> None:
        if dataset not in NAS201_DATASETS:
            raise ValueError("unknown NAS-Bench-201 dataset %r" % dataset)
        path = path or os.path.join(hpo_dir(), "nasbench201", dataset + ".pkl")
        self.table = _load_table(path, NAS201_SCHEMA)

    @staticmethod
    def params() -> List[Param]:
        return [Param.categorical(edge, NAS201_OPERATIONS) for edge in NAS201_EDGES]

    def evaluate(self, config: Dict[str, Any]) -> Tuple[float, float]:
        """Return ``(validation error in percent, cost in seconds)``."""
        error, cost = self.table[nas201_id(config)]
        return error, cost


# ----------------------------------------------------------------------------
# NAS-Bench-101 (HPOBench variants A, B and C)
# ----------------------------------------------------------------------------

NAS101_OPERATIONS = ("conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3")
_NAS101_LABEL_ORDER = ("conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3")
VERTICES = 7
MAX_EDGES = 9
EDGE_SLOTS = VERTICES * (VERTICES - 1) // 2  # 21 upper-triangular entries


def nas101_edge_pairs(variant: str) -> List[Tuple[int, int]]:
    """(source, target) meaning of each edge parameter, in parameter order."""
    if variant == "B":
        # B's selectors address the reversed bit string, decoded column-wise.
        return [pair for _, pair in sorted(
            (20 - (row + column * (column - 1) // 2), (row, column))
            for column in range(VERTICES) for row in range(column))]
    return [(row, column) for row in range(VERTICES) for column in range(row + 1, VERTICES)]


def _adjacency(config: Dict[str, Any], variant: str) -> List[List[int]]:
    matrix = [[0] * VERTICES for _ in range(VERTICES)]
    if variant == "A":
        for index, (row, column) in enumerate(nas101_edge_pairs("A")):
            matrix[row][column] = int(config["edge_%d" % index])
    elif variant == "B":
        for index in range(MAX_EDGES):
            row, column = nas101_edge_pairs("B")[int(config["edge_%d" % index])]
            matrix[row][column] = 1
    else:
        # Keep the ``num_edges`` highest-priority edges. Ties are broken the
        # way HPOBench does (numpy quicksort, reversed).
        import numpy as np
        priorities = np.asarray([float(config["edge_%d" % i]) for i in range(EDGE_SLOTS)])
        selected = {int(i) for i in np.argsort(priorities, kind="quicksort")[::-1][:int(config["num_edges"])]}
        for index, (row, column) in enumerate(nas101_edge_pairs("C")):
            matrix[row][column] = int(index in selected)
    return matrix


def _prune(matrix: List[List[int]], ops: Sequence[str]) -> Optional[Tuple[List[List[int]], List[str]]]:
    """Drop vertices that are not on an input-to-output path."""
    def reach(start: int, forward: bool) -> set:
        seen, frontier = {start}, [start]
        while frontier:
            node = frontier.pop()
            for other in range(VERTICES):
                edge = matrix[node][other] if forward else matrix[other][node]
                if edge and other not in seen:
                    seen.add(other)
                    frontier.append(other)
        return seen

    keep = sorted(reach(0, True) & reach(VERTICES - 1, False))
    if len(keep) < 2:
        return None
    return [[matrix[r][c] for c in keep] for r in keep], [ops[i] for i in keep]


def _module_hash(matrix: List[List[int]], labels: Sequence[int]) -> str:
    """NAS-Bench-101's isomorphism-invariant graph hash."""
    n = len(matrix)
    in_degree = [sum(matrix[r][c] for r in range(n)) for c in range(n)]
    out_degree = [sum(row) for row in matrix]
    hashes = [hashlib.md5(str(item).encode()).hexdigest() for item in zip(out_degree, in_degree, labels)]
    for _ in range(n):
        hashes = [
            hashlib.md5(("".join(sorted(hashes[r] for r in range(n) if matrix[r][v])) + "|"
                         + "".join(sorted(hashes[c] for c in range(n) if matrix[v][c])) + "|"
                         + hashes[v]).encode()).hexdigest()
            for v in range(n)]
    return hashlib.md5(str(sorted(hashes)).encode()).hexdigest()


def nas101_hash(config: Dict[str, Any], variant: str) -> Optional[str]:
    """Table key for a configuration, or ``None`` for an invalid graph."""
    matrix = _adjacency(config, variant)
    if sum(map(sum, matrix)) > MAX_EDGES:
        return None
    ops = ["input"] + [str(config["op_node_%d" % i]) for i in range(5)] + ["output"]
    pruned = _prune(matrix, ops)
    if pruned is None:
        return None
    matrix, ops = pruned
    labels = [-1] + [_NAS101_LABEL_ORDER.index(op) for op in ops[1:-1]] + [-2]
    return _module_hash(matrix, labels)


class NasBench101:
    fidelity = {"budget": 108}

    def __init__(self, variant: str, path: Optional[str] = None) -> None:
        if variant not in ("A", "B", "C"):
            raise ValueError("NAS-Bench-101 variant must be A, B or C")
        self.variant = variant
        self.table = _load_table(path or os.path.join(hpo_dir(), "nasbench101.pkl"), NAS101_SCHEMA)

    def params(self) -> List[Param]:
        params = [Param.categorical("op_node_%d" % i, NAS101_OPERATIONS) for i in range(5)]
        if self.variant == "A":
            params += [Param.categorical("edge_%d" % i, (0, 1)) for i in range(EDGE_SLOTS)]
        elif self.variant == "B":
            params += [Param.categorical("edge_%d" % i, tuple(range(EDGE_SLOTS))) for i in range(MAX_EDGES)]
        else:
            params.append(Param.integer("num_edges", 0, MAX_EDGES))
            params += [Param.real("edge_%d" % i, 0.0, 1.0) for i in range(EDGE_SLOTS)]
        return params

    def evaluate(self, config: Dict[str, Any]) -> Tuple[float, float]:
        """Return ``(validation error in [0, 1], cost in seconds)``; invalid graphs score error 1."""
        key = nas101_hash(config, self.variant)
        row = self.table.get(key) if key is not None else None
        if row is None:
            return 1.0, 0.0
        return row[0], row[1]


def _load_table(path: str, schema: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError("%s not found; run `python scripts/download_data.py --only nasbench101,nasbench201`" % path)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema") != schema:
        raise ValueError("%s has an unexpected format; re-run scripts/download_data.py" % path)
    return payload["table"]
