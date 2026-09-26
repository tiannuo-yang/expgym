import json
import os

from expgym.cli import main


def test_run_pool_and_summarize(tmp_path, capsys):
    output = str(tmp_path / "runs")
    assert main(["run", "--task", "toy", "--backend", "scripted", "--repeats", "2", "--output", output]) == 0
    files = sorted(os.listdir(os.path.join(output, "scripted", "toy", "tight")))
    assert files == ["toy__r0.json", "toy__r1.json"]
    with open(os.path.join(output, "scripted", "toy", "tight", "toy__r0.json")) as handle:
        record = json.load(handle)
    assert record["run"]["regime"] == "tight" and "gap" in record["score"]

    # Re-running skips finished jobs.
    capsys.readouterr()
    assert main(["run", "--task", "toy", "--backend", "scripted", "--repeats", "2", "--output", output]) == 0
    assert "6 already done, 0 to run" in capsys.readouterr().out

    assert main(["pool", "--task", "toy", "--backend", "scripted", "--agents", "2",
                 "--regimes", "tight", "--output", output]) == 0
    assert os.path.exists(os.path.join(output, "scripted", "toy", "tight", "poolact_n2", "toy__r0.json"))

    capsys.readouterr()
    assert main(["summarize", output]) == 0
    table = capsys.readouterr().out
    assert "## Single agent" in table and "## Agent pools" in table
    assert "| scripted | tight | poolact_n2 | " in table and "toy/gap" in table


def test_items_accept_groups_and_ids():
    from conftest import CounterTask
    from expgym.cli import select_items

    class GroupedTask(CounterTask):
        def items(self):
            return ["a1", "a2", "b1"]

        def item_group(self, item):
            return item[0]

    task = GroupedTask()
    assert select_items(task, "a,b1") == ["a1", "a2", "b1"]
    assert select_items(task, "1:") == ["a2", "b1"]


def test_custom_task_via_import(tmp_path, capsys):
    example = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "examples", "custom_environment.py")
    output = str(tmp_path / "runs")
    assert main(["run", "--import", example, "--task", "assay", "--backend", "openai", "--model", "m",
                 "--base-url", "http://127.0.0.1:9", "--max-retries", "0", "--timeout", "1",
                 "--items", "0", "--regimes", "tight", "--output", output]) == 1  # no server: job fails
    assert "FAILED" in capsys.readouterr().err
