"""Aggregate result files into the tables reported in the paper.

Scores are averaged over repeats within each item, then over items within each
group (tuning: benchmark family; search: whois / whatis), then over groups.
A task's AVG is the mean of its headline metrics (audit: label and exact
evidence-set accuracy), and Overall AVG is the mean of the task AVGs.
Accuracy and F1 are reported as percentages.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

HEADLINE = {"single": {"tuning": ["gap"], "toy": ["gap"], "search": ["f1"],
                       "audit": ["label_acc", "evidence_acc"]},
            "pool": {"tuning": ["gap"], "toy": ["gap"], "search": ["f1"], "audit": ["evidence_acc"]}}
PERCENT = {"f1", "label_acc", "evidence_acc"}
TASK_ORDER = ["tuning", "search", "audit"]  # Overall AVG averages these three


def _headline(kind: str, task: str) -> List[str]:
    if task in HEADLINE[kind]:
        return HEADLINE[kind][task]
    from expgym.envs import get_task  # custom tasks report their primary metric
    return [get_task(task).primary_metric()]


def load_results(root: str) -> List[Dict[str, Any]]:
    records = []
    for directory, _, files in os.walk(root):
        for name in sorted(files):
            if name.endswith(".json"):
                with open(os.path.join(directory, name), encoding="utf-8") as handle:
                    record = json.load(handle)
                if isinstance(record, dict) and "run" in record:
                    records.append(record)
    return records


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _metrics(record: Dict[str, Any]) -> Dict[str, float]:
    scores = record["score"] if record["run"]["kind"] == "single" else record["aggregate"]
    return {k: float(v) for k, v in scores.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def aggregate_task(records: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, float]:
    """Mean over repeats -> items -> groups for each metric."""
    per_item: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    for record in records:
        per_item[(record["run"]["group"], record["run"]["item"])].append(_metrics(record))
    result = {}
    for metric in metrics:
        groups: Dict[str, List[float]] = defaultdict(list)
        for (group, _), runs in per_item.items():
            values = [run[metric] for run in runs if metric in run]
            if values:
                groups[group].append(_mean(values))
        if groups:
            scale = 100.0 if metric in PERCENT else 1.0
            result[metric] = scale * _mean([_mean(v) for v in groups.values()])
    return result


def summarize(root: str, fmt: str = "markdown") -> str:
    records = load_results(root)
    if not records:
        return "no result files found under %s" % root
    table: Dict[Tuple[str, ...], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        run = record["run"]
        pool = ("%s_n%d" % (run["strategy"], run["agents"]),) if run["kind"] == "pool" else ()
        key = (run["model"], run["regime"]) + pool
        table[key][run["task"]].append(record)

    rows = []
    for key in sorted(table):
        kind = "pool" if len(key) == 3 else "single"
        row: Dict[str, Any] = {"kind": kind, "model": key[0], "regime": key[1]}
        if kind == "pool":
            row["strategy"] = key[2]
        averages = []
        tasks = sorted(table[key], key=lambda t: (TASK_ORDER.index(t) if t in TASK_ORDER else 99, t))
        for task in tasks:
            task_records = table[key][task]
            metrics = _headline(kind, task)
            values = aggregate_task(task_records, metrics)
            for metric, value in values.items():
                row["%s/%s" % (task, metric)] = value
            if values:
                row["%s/AVG" % task] = _mean(list(values.values()))
                if task in TASK_ORDER:
                    averages.append(row["%s/AVG" % task])
            row["%s/n" % task] = len(task_records)
        if len(averages) == len(TASK_ORDER):
            row["Overall AVG"] = _mean(averages)
        rows.append(row)

    if fmt == "json":
        return json.dumps(rows, indent=2)
    sections = []
    for kind, title in (("single", "## Single agent"), ("pool", "## Agent pools")):
        selected = [row for row in rows if row["kind"] == kind]
        if selected:
            sections.append(title + "\n\n" + _markdown(selected))
    return "\n\n".join(sections)


def _markdown(rows: List[Dict[str, Any]]) -> str:
    columns: List[str] = []
    for row in rows:
        columns += [c for c in row if c != "kind" and c not in columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = [("%.2f" % row[c]) if isinstance(row.get(c), float) else str(row.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
