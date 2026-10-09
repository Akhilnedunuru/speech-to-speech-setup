"""Tests for brain.py — mocked LLM, no network, no pipecat needed.

The LangGraph graph is exercised through AgenticBrain with scripted
llm_fn mocks:
  - direct response (no tools)
  - one tool round then response
  - tool loop hitting the 3-round cap
  - unknown tool handled gracefully
"""

import asyncio

from brain import MAX_ROUNDS, AgenticBrain, build_graph, get_weather


def scripted_llm(script):
    """llm_fn that pops outputs from a script list. Falls back to a
    canned response when the script is exhausted (e.g. the cap wrap-up)."""
    calls = []

    def fn(messages):
        calls.append([m.get("content", "")[:40] for m in messages])
        if script:
            return script.pop(0)
        return {"type": "response", "text": "fallback answer"}
    fn.calls = calls
    return fn


def test_direct_response_no_tools():
    llm = scripted_llm([{"type": "response", "text": "Hello there!"}])
    brain = AgenticBrain(llm_fn=llm)
    out = brain.run("hi")
    assert out == "Hello there!", out
    assert len(llm.calls) == 1, llm.calls
    print("PASS: test_direct_response_no_tools")


def test_one_tool_round():
    llm = scripted_llm([
        {"type": "tool", "name": "get_weather", "args": {"city": "Dallas"}},
        {"type": "response", "text": "It's 72F and sunny in Dallas."},
    ])
    brain = AgenticBrain(llm_fn=llm)
    out = brain.run("weather in Dallas?")
    assert out == "It's 72F and sunny in Dallas.", out
    assert len(llm.calls) == 2, len(llm.calls)
    # The tool result must have been fed back into the second call.
    second_call_text = " ".join(llm.calls[1])
    assert "72F" in second_call_text, second_call_text
    print("PASS: test_one_tool_round")


def test_cap_respected_at_max_rounds():
    # LLM always wants another tool call; the graph must stop at MAX_ROUNDS
    # and still produce a final response via the wrap-up call.
    # Script has exactly 4 tool calls: 3 real rounds + the 4th think that
    # triggers the cap. The wrap-up then hits the empty-script fallback.
    llm = scripted_llm(
        [{"type": "tool", "name": "get_weather", "args": {"city": f"City{i}"}}
         for i in range(4)]
    )
    brain = AgenticBrain(llm_fn=llm, max_rounds=MAX_ROUNDS)
    out = brain.run("weather everywhere?")
    # 4 think calls (the 4th hits the cap) + 1 wrap-up call = 5 total.
    # Exactly MAX_ROUNDS tool executions happened.
    assert len(llm.calls) == MAX_ROUNDS + 2, len(llm.calls)
    assert out == "fallback answer", out
    # The wrap-up call must carry the no-more-tools instruction.
    wrap_text = " ".join(llm.calls[-1])
    assert "No more tool calls" in wrap_text, wrap_text
    print("PASS: test_cap_respected_at_max_rounds")


def test_unknown_tool_graceful():
    llm = scripted_llm([
        {"type": "tool", "name": "nope_not_real", "args": {}},
        {"type": "response", "text": "I can't do that one."},
    ])
    brain = AgenticBrain(llm_fn=llm)
    out = brain.run("do the impossible")
    assert out == "I can't do that one.", out
    assert "Unknown tool" in " ".join(llm.calls[1]), llm.calls[1]
    print("PASS: test_unknown_tool_graceful")


def test_tool_exception_graceful():
    def boom(city: str):
        raise RuntimeError("kaput")

    llm = scripted_llm([
        {"type": "tool", "name": "boom", "args": {"city": "X"}},
        {"type": "response", "text": "Tool broke, moving on."},
    ])
    brain = AgenticBrain(llm_fn=llm, tools={"boom": boom})
    out = brain.run("break it")
    assert out == "Tool broke, moving on.", out
    print("PASS: test_tool_exception_graceful")


def test_get_weather_mock():
    assert "Dallas" in get_weather("Dallas")
    print("PASS: test_get_weather_mock")


def test_graph_rounds_tracked():
    llm = scripted_llm([
        {"type": "tool", "name": "get_weather", "args": {"city": "A"}},
        # wrap-up (cap=1, so the 2nd think triggers it)
    ])

    from brain import _initial_state
    graph = build_graph(llm, max_rounds=1)
    result = graph.invoke(_initial_state("hi"))
    assert result["rounds"] == 1, result["rounds"]
    assert len(result["tool_calls"]) == 1
    name, args, res = result["tool_calls"][0]
    assert name == "get_weather" and "A" in res
    print("PASS: test_graph_rounds_tracked")


if __name__ == "__main__":
    test_direct_response_no_tools()
    test_one_tool_round()
    test_cap_respected_at_max_rounds()
    test_unknown_tool_graceful()
    test_tool_exception_graceful()
    test_get_weather_mock()
    test_graph_rounds_tracked()
    print("\nAll brain tests passed.")
