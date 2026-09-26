"""Multi-hop search over the PhantomWiki corpus.

The agent answers 3-hop ``whois`` / ``whatis`` questions with a single
``search`` tool that returns one full article: an exact (case-insensitive)
title match if possible, otherwise the best keyword match. The first
retrieval of an article costs 280-320 simulated seconds; any later query that
returns an already retrieved article is free. Answers are scored with set F1
over entity names.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

from expgym.envs.base import Environment, Task, ToolInputError, ToolResult, function_tool, stable_uniform
from expgym.paths import phantom_wiki_dir

# PhantomWiki question types: 11/12 are 3-hop "whois", 27/28 are 3-hop "whatis".
QUESTION_TYPES = {11: "whois", 12: "whois", 27: "whatis", 28: "whatis"}
MAX_ANSWERS = 20
DEFAULT_SEEDS = (2, 3)  # 39 whois + 34 whatis questions
COST_RANGE = (280.0, 320.0)
_TOKEN = re.compile(r"\b\w+\b")

PROMPT = """You are answering a question by searching a wiki of fictional characters.

Tool:
- search: look up information in the wiki.
  Input: {{"query": "Person Name"}} or {{"query": "keyword phrase"}}
  If the query exactly matches an article title, that article is returned.
  Otherwise, the most relevant article is returned based on keyword matching.

Strategy:
1. The wiki contains articles about fictional people, each with Family, Friends, and Attributes sections.
2. Questions ask about relationship chains (e.g., 'Who is the friend of the parent of X?').
3. Start from the person named (or described) in the question, then follow the chain step by step.
4. Each search returns one full article. Re-searching a previously returned article is free.
5. Your answer should list ALL matching names, separated by commas.

Native tool invocation:
  Call the search function with arguments {{"query": "Person Name"}} using a native tool call.
  Answer: Name1, Name2, ...

Question: {question}"""

TOOL = function_tool(
    "search",
    "Search the fictional-character wiki by person name or keyword phrase. "
    "Return one complete article and its simulated cost; previously returned articles are free.",
    {"type": "object",
     "properties": {"query": {"type": "string", "minLength": 1,
                              "description": "A person name or keyword phrase to search for."}},
     "required": ["query"], "additionalProperties": False},
)


def _tokens(text: str) -> Set[str]:
    return {token.lower() for token in _TOKEN.findall(text)}


def parse_names(answer: Optional[str]) -> Set[str]:
    """Normalize an answer into a set of lower-cased names."""
    if not answer or not answer.strip():
        return set()
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, list):
            return {" ".join(str(n).lower().split()) for n in parsed if str(n).strip()}
    except ValueError:
        pass
    names = set()
    for part in re.split(r"[,;\n]", answer):
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", part).strip().strip("-*•").strip()
        if 1 <= len(cleaned.split()) <= 5:
            names.add(" ".join(cleaned.lower().split()))
    return names


def set_f1(predicted: Set[str], gold: Set[str]) -> float:
    if not gold:
        return float(not predicted)
    overlap = len(predicted & gold)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


class Corpus:
    def __init__(self, articles: Dict[str, str]) -> None:
        self.articles = articles
        self.by_lower = {title.lower(): title for title in articles}
        self.title_tokens = {title: _tokens(title) for title in articles}
        self.body_tokens = {title: _tokens(text) for title, text in articles.items()}

    def lookup(self, query: str) -> Optional[str]:
        title = self.by_lower.get(query.lower())
        if title is not None:
            return title
        query_tokens = _tokens(query)
        best, best_score = None, 0
        for title in self.articles:
            score = (5 * len(query_tokens & self.title_tokens[title])
                     + len(query_tokens & self.body_tokens[title]))
            if score > best_score:
                best, best_score = title, score
        return best


class SearchEnv(Environment):
    reference_cost = 300.0

    def __init__(self, corpus: Corpus, question: str, answers: Sequence[str]) -> None:
        self.corpus = corpus
        self.question = question
        self.gold = {" ".join(a.lower().split()) for a in answers}
        self.seen: Set[str] = set()

    def task_prompt(self) -> str:
        return PROMPT.format(question=self.question)

    def tools(self):
        return [TOOL]

    def call(self, name, arguments):
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolInputError("No query provided.")
        title = self.corpus.lookup(query)
        if title is None:
            return ToolResult("No matching article found.", 0.0)
        text = "Article: %s\n\n%s" % (title, self.corpus.articles[title])
        if title in self.seen:
            return ToolResult(text, 0.0)
        self.seen.add(title)
        return ToolResult(text, stable_uniform("search:" + title, *COST_RANGE))

    def on_reuse(self, name, arguments, result):
        if result.text.startswith("Article: "):  # later retrievals of this article are free
            self.seen.add(result.text.split("\n", 1)[0][len("Article: "):])

    def score(self, answer, observations):
        return {"f1": set_f1(parse_names(answer), self.gold)}

    def describe_action(self, name, arguments):
        return 'search "%s"' % str(arguments.get("query", ""))[:60]

    def describe_outcome(self, name, arguments, result):
        title = result.text.split("\n", 1)[0][len("Article: "):] if result.text.startswith("Article: ") else ""
        label = 'search "%s"' % arguments.get("query", "")
        return label + (' -> "%s"' % title if title else "")

    def coverage_note(self, outcomes):
        return "%d queries explored so far.\n  Try a query not listed above." % len(outcomes)


class SearchTask(Task):
    """Items are ``seed<k>/<index>`` over the filtered questions of each seed."""

    name = "search"
    pool_aggregation = "vote"

    def __init__(self, seeds: Sequence[int] = DEFAULT_SEEDS, data_dir: Optional[str] = None) -> None:
        self.seeds = tuple(seeds)
        self.data_dir = data_dir or phantom_wiki_dir()
        self._corpora: Dict[int, Corpus] = {}
        self._questions: Dict[int, List[dict]] = {}

    def _parquet(self, kind: str, seed: int, columns: List[str]) -> List[dict]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("the search task needs pyarrow (pip install pyarrow)") from exc
        path = os.path.join(self.data_dir, kind, "depth_20_size_5000_seed_%d-00000-of-00001.parquet" % seed)
        if not os.path.exists(path):
            raise FileNotFoundError("%s not found; run `python scripts/download_data.py --only search`" % path)
        return pq.read_table(path, columns=columns).to_pylist()

    def questions(self, seed: int) -> List[dict]:
        if seed not in self._questions:
            rows = self._parquet("question-answer", seed, ["question", "answer", "type", "difficulty"])
            rows = [r for r in rows if r["type"] in QUESTION_TYPES and len(r["answer"]) <= MAX_ANSWERS]
            rows.sort(key=lambda r: (r["type"], r["difficulty"]))
            self._questions[seed] = rows
        return self._questions[seed]

    def corpus(self, seed: int) -> Corpus:
        if seed not in self._corpora:
            rows = self._parquet("text-corpus", seed, ["title", "article"])
            self._corpora[seed] = Corpus({r["title"]: r["article"] for r in rows})
        return self._corpora[seed]

    def _locate(self, item: str) -> Tuple[int, dict]:
        seed, index = item.split("/")
        seed_number = int(seed[len("seed"):])
        return seed_number, self.questions(seed_number)[int(index)]

    def items(self):
        return ["seed%d/%d" % (seed, i) for seed in self.seeds for i in range(len(self.questions(seed)))]

    def make_env(self, item, repeat=0):
        seed, row = self._locate(item)
        return SearchEnv(self.corpus(seed), row["question"], row["answer"])

    def item_group(self, item):
        return QUESTION_TYPES[self._locate(item)[1]["type"]]

    def primary_metric(self):
        return "f1"

    def vote(self, answers):
        """Majority vote over normalized name sets; ties favour non-empty, then earlier answers."""
        keys = [tuple(sorted(parse_names(a))) for a in answers]
        counts = Counter(keys)
        winner = max(counts, key=lambda k: (counts[k], bool(k), -keys.index(k)))
        return answers[keys.index(winner)]
