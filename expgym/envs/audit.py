"""Evidence audit on ContractNLI.

For each of 17 hypotheses about a contract, the agent assigns a label
(Entailment / Contradiction / NotMentioned) and the set of supporting segment
IDs. The ``human_feedback`` tool verifies a proposed evidence set for one
hypothesis and replies ``Evidence Correct``, ``Evidence Incomplete`` (with a
coarse hint describing what is missing) or ``Contain Irrelevant Evidences``.
Every call costs 280-320 simulated seconds, including repeated requests.
Answers are scored by label accuracy and exact evidence-set accuracy.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from expgym.envs.base import (Environment, Task, ToolInputError, ToolResult, function_tool,
                              stable_uniform)
from expgym.paths import contract_nli_file, resource

LABELS = ("Entailment", "Contradiction", "NotMentioned")
DEFAULT_DOCUMENTS = 13  # the first 13 documents of the ContractNLI test split
COST_RANGE = (280.0, 320.0)

INSTRUCTIONS = """You are auditing a legal document against {n} hypotheses.
Your goal: maximize the correctness of ALL labels and evidence IDs in your final answer.
For each hypothesis, determine: Entailment, Contradiction, or NotMentioned.
For Entailment/Contradiction, identify the exact evidence segment IDs.

Tool: human_feedback — verify your proposed evidence for a hypothesis. Can be called multiple times.
  Input: {{{{"nda_id": "...", "evidence_ids": [segment_ids]}}}}

Use human feedback to improve the correctness of your labels and evidence IDs.

Call the human_feedback function with arguments {{"nda_id": "nda-11", "evidence_ids": [3, 7]}} using a native tool call.
Answer: {{"nda-11": {{"label": "...", "evidence_ids": [...]}}, ...}}
Your final answer MUST include ALL {n} hypotheses.

Document segments:
{segments}

Hypotheses:
{hypotheses}"""


def parse_audit_answer(answer: Optional[str]) -> Dict[str, Any]:
    """Parse a JSON object answer, optionally wrapped in a Markdown code fence."""
    if not answer:
        return {}
    text = answer.strip().rstrip(";").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def normalize_label(value: Any) -> str:
    """Map common spellings (e.g. "entailed", "neutral") to the task's labels."""
    key = re.sub(r"[^a-z]", "", str(value or "").lower())
    aliases = {"entailment": "Entailment", "entailed": "Entailment", "contradiction": "Contradiction",
               "contradicted": "Contradiction", "notmentioned": "NotMentioned", "neutral": "NotMentioned"}
    return aliases.get(key, str(value or "").strip())


def evidence_set(value: Any) -> frozenset:
    try:
        return frozenset(int(v) for v in value)
    except (TypeError, ValueError):
        return frozenset()


class AuditEnv(Environment):
    reference_cost = 300.0

    def __init__(self, document: dict, hypotheses: Dict[str, dict],
                 order: Sequence[str], hints: Dict[int, str]) -> None:
        self.document = document
        self.doc_id = int(document["id"])
        self.gold = document["annotation_sets"][0]["annotations"]
        self.hypotheses = hypotheses
        self.order = [h for h in order if h in hypotheses]
        self.hints = hints

    def task_prompt(self):
        segments = "\n".join(
            "[%d] %s" % (i, re.sub(r"\s+", " ", text).strip())
            for i, text in enumerate(self._segments()))
        hypotheses = "\n".join(
            "- %s: %s | %s" % (h, self.hypotheses[h].get("short_description", ""),
                               self.hypotheses[h].get("hypothesis", ""))
            for h in self.order)
        return INSTRUCTIONS.format(n=len(self.order), segments=segments, hypotheses=hypotheses)

    def _segments(self) -> List[str]:
        text = self.document["text"]
        return [text[int(start):int(end)] for start, end in self.document["spans"]]

    def tools(self):
        return [function_tool(
            "human_feedback",
            "Verify the proposed evidence segment IDs for a document hypothesis. "
            "Return evidence feedback and simulated cost, not the correct entailment label.",
            {"type": "object",
             "properties": {
                 "nda_id": {"type": "string", "enum": list(self.gold)},
                 "evidence_ids": {"type": "array", "items": {"type": "integer"},
                                  "description": "Proposed document evidence segment IDs; [] for none."}},
             "required": ["nda_id", "evidence_ids"], "additionalProperties": False})]

    def call(self, name, arguments):
        nda_id = str(arguments.get("nda_id", "")).strip()
        if nda_id not in self.gold:
            raise ToolInputError("Unknown nda_id: %s" % nda_id)
        raw = arguments.get("evidence_ids") or []
        if not isinstance(raw, list):
            raise ToolInputError("evidence_ids must be a list of integers")
        try:
            proposed = {int(v) for v in raw}
        except (TypeError, ValueError):
            raise ToolInputError("evidence_ids must be a list of integers")
        cost = stable_uniform("audit:%d:%s:%s" % (self.doc_id, nda_id, ",".join(map(str, sorted(proposed)))),
                              *COST_RANGE)
        gold = self.gold[nda_id]
        spans = {int(s) for s in gold.get("spans", [])}
        if gold["choice"] == "NotMentioned":
            verdict = "Contain Irrelevant Evidences" if proposed else "Evidence Correct"
        elif proposed == spans:
            verdict = "Evidence Correct"
        elif proposed <= spans:
            verdict = "Evidence Incomplete | missing: " + self._hint(spans - proposed)
        else:
            verdict = "Contain Irrelevant Evidences"
        return ToolResult(verdict, cost)

    def _hint(self, missing) -> str:
        parts = sorted({self.hints[s] for s in missing if s in self.hints})
        return ", ".join(parts) if parts else "<none>"

    def score(self, answer, observations):
        parsed = parse_audit_answer(answer)
        label_hits = evidence_hits = 0
        for nda_id in self.order:
            entry = parsed.get(nda_id)
            if not isinstance(entry, dict):
                continue
            gold = self.gold[nda_id]
            label_hits += entry.get("label") == gold["choice"]
            evidence_hits += evidence_set(entry.get("evidence_ids", [])) == \
                frozenset(int(s) for s in gold.get("spans", []))
        n = len(self.order)
        return {"label_acc": label_hits / n, "evidence_acc": evidence_hits / n}

    def describe_action(self, name, arguments):
        return "verify nda:%s ev:%s" % (arguments.get("nda_id", "?"), arguments.get("evidence_ids", []))

    def describe_outcome(self, name, arguments, result):
        return "nda:%s  ev:%s  [%s]" % (arguments.get("nda_id", "?"),
                                        arguments.get("evidence_ids", []), result.text)

    def coverage_note(self, outcomes):
        attempted = {o.arguments.get("nda_id") for o in outcomes}
        return "%d NDAs attempted so far." % len(attempted)


class AuditTask(Task):
    """Items are document indices; ``repeat`` selects the hypothesis ordering."""

    name = "audit"
    pool_aggregation = "vote"

    def __init__(self, documents: int = DEFAULT_DOCUMENTS, path: Optional[str] = None) -> None:
        self.documents = documents
        self.path = path or contract_nli_file()
        self._data: Optional[dict] = None
        with open(resource("audit_hypothesis_orders.json"), encoding="utf-8") as handle:
            self.orders = json.load(handle)["orders"]
        with open(resource("contract_nli_hints.json"), encoding="utf-8") as handle:
            self.hints = {int(item["doc_id"]): {int(k): v for k, v in item["span_to_dimension"].items()}
                          for item in json.load(handle)}

    @property
    def data(self) -> dict:
        if self._data is None:
            if not os.path.exists(self.path):
                raise FileNotFoundError("%s not found; run `python scripts/download_data.py --only audit`"
                                        % self.path)
            with open(self.path, encoding="utf-8") as handle:
                self._data = json.load(handle)
        return self._data

    def items(self):
        return [str(i) for i in range(self.documents)]

    def make_env(self, item, repeat=0):
        document = self.data["documents"][int(item)]
        order = self.orders[repeat % len(self.orders)]
        return AuditEnv(document, self.data["labels"], order, self.hints.get(int(document["id"]), {}))

    def primary_metric(self):
        return "evidence_acc"

    def vote(self, answers):
        """Per hypothesis: majority label, then majority evidence set among that label."""
        parsed = [parse_audit_answer(a) for a in answers]
        voted = {}
        for nda_id in sorted({h for p in parsed for h in p}):
            entries = [p[nda_id] for p in parsed if isinstance(p.get(nda_id), dict)]
            if not entries:
                continue
            labels = [normalize_label(e.get("label")) for e in entries]
            label = Counter(labels).most_common(1)[0][0]
            evidence = Counter(tuple(sorted(evidence_set(entry.get("evidence_ids", []))))
                               for entry, entry_label in zip(entries, labels)
                               if entry_label == label).most_common(1)[0][0]
            voted[nda_id] = {"label": label, "evidence_ids": list(evidence)}
        return json.dumps(voted, sort_keys=True)
