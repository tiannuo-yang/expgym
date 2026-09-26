import json

from expgym.agent import REGIMES, extract_answer, get_regime, run_agent, _fit_context
from expgym.llm import ModelTurn, ScriptedBackend, ToolCall


class RecordingBackend(ScriptedBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_choices = []

    def generate(self, messages, tools=None, tool_choice="auto"):
        self.tool_choices.append(tool_choice)
        return super().generate(messages, tools, tool_choice)


def tool_messages(result):
    return [m["content"] for m in result["messages"] if m["role"] == "tool"]


def test_answer_without_tools(counter_env):
    result = run_agent(ScriptedBackend([], "42"), counter_env, REGIMES["free"])
    assert result["answer"] == "42"
    assert result["termination"] == "answered"
    assert result["score"]["correct"] == 1.0
    assert result["feedback_cost"] == 0.0


def test_free_regime_hides_cost(counter_env):
    backend = ScriptedBackend([("probe", {"x": 1}), ("probe", {"x": 2})], "42")
    result = run_agent(backend, counter_env, REGIMES["free"])
    assert tool_messages(result) == ["Observation: value=2", "Observation: value=4"]
    assert result["feedback_cost"] == 200.0


def test_tight_budget_withholds_boundary_observation(counter_env):
    backend = RecordingBackend([("probe", {"x": i}) for i in range(10)], "42")
    result = run_agent(backend, counter_env, REGIMES["tight"])  # B = 3 * 100
    assert result["budget"] == 300.0
    assert result["termination"] == "budget_exhausted"
    messages = tool_messages(result)
    assert messages[0] == "Observation: value=0 | cost=100s [time_left=200s]"
    assert messages[1] == "Observation: value=2 | cost=100s [time_left=100s]"
    assert "withheld" in messages[2]
    assert result["score"]["seen"] == 2  # the withheld result is not visible
    assert [c["visible"] for c in result["tool_calls"]] == [True, True, False]
    assert backend.tool_choices[-1] == "none"  # forced final answer
    assert result["answer"] == "42"


def test_step_limit_counts_every_turn(counter_env):
    backend = ScriptedBackend([("probe", {"x": i}) for i in range(10)], "42")
    result = run_agent(backend, counter_env, REGIMES["free"], max_steps=4)
    assert result["termination"] == "step_limit"
    assert len(result["tool_calls"]) == 4
    assert result["steps"] == 5  # four decisions plus the forced answer


def test_tool_input_errors_are_free_observations(counter_env):
    backend = ScriptedBackend([("probe", {"x": "a"})], "42")
    result = run_agent(backend, counter_env, REGIMES["tight"])
    assert tool_messages(result) == ["Observation: Tool error: x must be an integer"]
    assert result["feedback_cost"] == 0.0


class MultiCallBackend:
    """Always emits two tool calls in one turn."""

    def generate(self, messages, tools=None, tool_choice="auto"):
        if tool_choice == "none":
            return ModelTurn(text="Answer: 7")
        calls = [ToolCall("a", "probe", json.dumps({"x": 1})), ToolCall("b", "probe", json.dumps({"x": 2}))]
        return ModelTurn(text="", tool_calls=calls)


def test_invalid_turns_then_forced_answer(counter_env):
    result = run_agent(MultiCallBackend(), counter_env, REGIMES["free"], max_invalid_turns=1)
    assert result["termination"] == "invalid_response"
    assert result["answer"] == "7"
    assert not result["tool_calls"]
    errors = tool_messages(result)
    assert len(errors) == 4 and all(e.startswith("Protocol error") for e in errors)


def test_extract_answer():
    assert extract_answer("Answer: Ada, Bob") == "Ada, Bob"
    assert extract_answer("<think>Answer: no</think>\n**Final Answer:** yes") == "yes"
    assert extract_answer("just text") == "just text"
    assert extract_answer("   ") is None


def test_custom_regime():
    regime = get_regime("beta=5")
    assert regime.limit(300.0) == 1500.0 and regime.show_cost
    assert REGIMES["free"].limit(300.0) is None


def test_context_fitting_keeps_tool_pairs():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(20):
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i)}]})
        messages.append({"role": "tool", "tool_call_id": str(i), "content": "x" * 2000})
    fitted = _fit_context(messages, 2000)
    assert fitted[:2] == messages[:2]
    assert fitted[2]["role"] == "assistant"
    assert fitted[-1] == messages[-1]
    assert len(fitted) < len(messages)
