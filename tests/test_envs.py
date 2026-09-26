import json
import os

import pytest

from expgym.envs.audit import AuditEnv, AuditTask, parse_audit_answer
from expgym.envs.base import Observation, ToolInputError
from expgym.envs.params import Param, validate_config
from expgym.envs.search import Corpus, SearchEnv, SearchTask, parse_names, set_f1
from expgym.envs.tuning import ToyTuningTask, error_to_perf, parse_config
from expgym.paths import hpo_dir


# ---------------------------------------------------------------- parameters

def test_validate_config():
    params = [Param.integer("layers", 1, 4), Param.real("lr", 1e-4, 1e-1, log=True),
              Param.categorical("act", ("relu", "tanh"))]
    assert validate_config(params, {"layers": 2.0, "lr": 0.01, "act": "relu"}) == \
        {"layers": 2, "lr": 0.01, "act": "relu"}
    with pytest.raises(ToolInputError, match="missing lr"):
        validate_config(params, {"layers": 2, "act": "relu"})
    with pytest.raises(ToolInputError, match="layers out of range"):
        validate_config(params, {"layers": 9, "lr": 0.01, "act": "relu"})
    with pytest.raises(ToolInputError, match="Unknown"):
        validate_config(params, {"layers": 2, "lr": 0.01, "act": "relu", "extra": 1})
    assert params[1].describe().startswith("- lr: 0.0001 to 0.1 (log) (default=0.00316")


# -------------------------------------------------------------------- tuning

def test_error_to_perf_handles_fraction_and_percent():
    assert error_to_perf(0.25) == 0.75
    assert error_to_perf(25.0) == 0.75
    assert error_to_perf(1.0) == 0.0


def test_toy_tuning_scoring_rules():
    env = ToyTuningTask().make_env("toy")
    config = {"num_layers": 6, "hidden_width": 512, "learning_rate": 3e-4, "dropout": 0.15, "batch_size": 64}
    result = env.call("evaluate_config", config)
    assert result.perf > 0 and result.cost > 0 and result.text.startswith("perf=")
    seen = [Observation("evaluate_config", config, result)]

    matched = env.score(json.dumps(config), seen)
    assert matched["scored_from"] == "matched_evaluation" and matched["perf"] == result.perf
    fallback = env.score('{"num_layers": 2}', seen)
    assert fallback["scored_from"] == "best_visible_evaluation"
    offline = env.score("```json\n%s\n```" % json.dumps(config), [])
    assert offline["scored_from"] == "offline_evaluation" and offline["perf"] == result.perf
    assert env.score(None, seen) == {"gap": 0.0, "perf": None, "scored_from": "none"}
    assert env.gap(env.reference["best_perf"]) == pytest.approx(100.0)
    assert env.gap(0.0) == 0.0


def test_parse_config():
    assert parse_config('{"a": 1};') == {"a": 1}
    assert parse_config("not json") is None


@pytest.mark.skipif(not os.path.exists(os.path.join(hpo_dir(), "nasbench201")), reason="NAS data not downloaded")
def test_nasbench_reference_optimum_is_reproduced():
    from expgym.envs.tuning import TuningTask
    task = TuningTask(["nasbench201/cifar100"])
    env = task.make_env("nasbench201/cifar100")
    reference = task.references["nasbench201/cifar100"]
    result = env.call("evaluate_config", reference["best_config"])
    assert result.perf == pytest.approx(reference["best_perf"])
    assert env.gap(result.perf) == pytest.approx(100.0)


# -------------------------------------------------------------------- search

def make_search_env():
    corpus = Corpus({"Ada Byron": "Ada's friend is Bob Stone.", "Bob Stone": "Bob likes chess."})
    return SearchEnv(corpus, "Who is the friend of Ada Byron?", ["Bob Stone"])


def test_search_costs_first_retrieval_only():
    env = make_search_env()
    first = env.call("search", {"query": "ada byron"})
    assert first.text.startswith("Article: Ada Byron") and 280 <= first.cost <= 320
    assert env.call("search", {"query": "Ada"}).cost == 0.0  # same article via a new query
    assert env.call("search", {"query": "chess"}).text.startswith("Article: Bob Stone")
    assert env.call("search", {"query": "zzz"}).text == "No matching article found."
    with pytest.raises(ToolInputError):
        env.call("search", {"query": " "})


def test_search_reused_article_is_not_charged_again():
    env = make_search_env()
    shared = make_search_env().call("search", {"query": "Ada Byron"})
    env.on_reuse("search", {"query": "Ada Byron"}, shared)
    assert env.call("search", {"query": "ada"}).cost == 0.0


def test_search_scoring_and_vote():
    env = make_search_env()
    assert env.score("Bob Stone", [])["f1"] == 1.0
    assert env.score("Bob Stone, Ada Byron", [])["f1"] == pytest.approx(2 / 3)
    assert parse_names('["Bob  Stone"]') == {"bob stone"}
    assert set_f1(set(), set()) == 1.0
    task = SearchTask()
    assert task.vote(["Bob, Ann", "ann, bob", "Carl", ""]) == "Bob, Ann"


# --------------------------------------------------------------------- audit

def make_audit_env():
    document = {"id": 7, "text": "Alpha. Beta. Gamma.", "spans": [[0, 6], [7, 12], [13, 19]],
                "annotation_sets": [{"annotations": {
                    "nda-1": {"choice": "Entailment", "spans": [0, 2]},
                    "nda-2": {"choice": "NotMentioned", "spans": []}}}]}
    hypotheses = {"nda-1": {"short_description": "a", "hypothesis": "A"},
                  "nda-2": {"short_description": "b", "hypothesis": "B"}}
    return AuditEnv(document, hypotheses, ["nda-2", "nda-1"], {2: "Gamma clause"})


def test_audit_feedback():
    env = make_audit_env()
    assert "[0] Alpha.\n[1] Beta.\n[2] Gamma." in env.task_prompt()
    assert env.call("human_feedback", {"nda_id": "nda-1", "evidence_ids": [0, 2]}).text == "Evidence Correct"
    partial = env.call("human_feedback", {"nda_id": "nda-1", "evidence_ids": [0]})
    assert partial.text == "Evidence Incomplete | missing: Gamma clause" and 280 <= partial.cost <= 320
    assert env.call("human_feedback", {"nda_id": "nda-1", "evidence_ids": [1]}).text == \
        "Contain Irrelevant Evidences"
    assert env.call("human_feedback", {"nda_id": "nda-2", "evidence_ids": []}).text == "Evidence Correct"
    with pytest.raises(ToolInputError):
        env.call("human_feedback", {"nda_id": "nda-9", "evidence_ids": []})


def test_audit_scoring_and_vote():
    env = make_audit_env()
    answer = {"nda-1": {"label": "Entailment", "evidence_ids": [2, 0]},
              "nda-2": {"label": "Entailment", "evidence_ids": []}}
    assert env.score(json.dumps(answer), []) == {"label_acc": 0.5, "evidence_acc": 1.0}
    assert parse_audit_answer("```json\n{\"a\": 1}\n```") == {"a": 1}
    other = {"nda-1": {"label": "Contradiction", "evidence_ids": [1]}}
    spelled = {"nda-1": {"label": "entailed", "evidence_ids": [0, 2]}}
    voted = json.loads(AuditTask().vote([json.dumps(answer), json.dumps(spelled), json.dumps(other)]))
    assert voted["nda-1"] == {"label": "Entailment", "evidence_ids": [0, 2]}


def test_packaged_resources_load():
    task = AuditTask()
    assert len(task.orders) == 3 and all(len(order) == 17 for order in task.orders)
    assert task.items() == [str(i) for i in range(13)]
