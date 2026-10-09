"""Phase 3, step 4: LangGraph reasoning loop for the LLM stage.

Replaces the single-shot Groq call with an agentic loop:

    think -> [tool?] -> act -> observe -> think -> ... -> respond

- `think`: LLM call. Returns either a tool call or a final response.
- `act`: executes the requested tool.
- `observe`: tool result is appended to messages; loop back to think.
- `respond`: terminal node, emits the final text.

Voice latency constraint: MAX_ROUNDS (default 3) caps tool iterations.
When the cap is hit mid-loop, the LLM gets one final "answer now" call
instead of another tool round. While the loop runs (round 2+), the
pipeline should play filler audio ("Let me look that up...") -- that
integration is a later step; here we just log "Tool loop round N".

The LLM is injected as `llm_fn(messages) -> dict` so the graph is fully
testable with mocks and adaptable to any backend later:
    {"type": "tool", "name": "get_weather", "args": {"city": "..."}}
    {"type": "response", "text": "..."}

Ships with one example tool (get_weather mock). Real tools come later.

Pipecat wrapper: BrainProcessor (FrameProcessor) takes user text
(TranscriptionFrame from STT, or TextFrame) and emits a TextFrame with the
final response. The pipecat import is guarded so the graph logic stays
testable without it.
"""

import logging
import operator
from typing import Annotated, Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger("pipecat-brain")

MAX_ROUNDS = 3

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def get_weather(city: str) -> str:
    """Mock weather tool. Real implementation comes later."""
    return f"Mock weather in {city}: 72F, sunny, light breeze."


TOOLS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
}


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class BrainState(TypedDict):
    """LangGraph state. messages/tool_calls append; the rest overwrite."""

    messages: Annotated[list, operator.add]
    rounds: int
    last_out: Optional[dict]
    final_response: Optional[str]
    tool_calls: Annotated[list, operator.add]


def _initial_state(user_text: str) -> dict:
    return {
        "messages": [{"role": "user", "content": user_text}],
        "rounds": 0,
        "last_out": None,
        "final_response": None,
        "tool_calls": [],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    llm_fn: Callable[[list[dict]], dict],
    tools: dict[str, Callable[..., str]] | None = None,
    max_rounds: int = MAX_ROUNDS,
):
    """Build the think -> act -> observe loop. Returns a compiled graph.

    llm_fn(messages) -> {"type": "tool", "name": str, "args": dict}
                     | {"type": "response", "text": str}
    """
    tools = tools or TOOLS

    def think(state: dict) -> dict:
        out = llm_fn(state["messages"])
        return {"last_out": out}

    def route(state: dict) -> str:
        out = state.get("last_out") or {}
        if out.get("type") == "tool" and state["rounds"] < max_rounds:
            return "act"
        return "respond"

    def act(state: dict) -> dict:
        out = state["last_out"]
        name, args = out["name"], out.get("args", {})
        tool = tools.get(name)
        if tool is None:
            result = f"Unknown tool: {name}"
        else:
            try:
                result = tool(**args)
            except Exception as e:
                result = f"Tool {name} failed: {e}"
        round_no = state["rounds"] + 1
        logger.info("Tool loop round %d: %s(%s)", round_no, name, args)
        return {
            "messages": [{"role": "tool", "name": name, "content": result}],
            "rounds": round_no,
            "tool_calls": [(name, args, result)],
        }

    def respond(state: dict) -> dict:
        out = state.get("last_out") or {}
        if out.get("type") == "response":
            return {"final_response": out["text"]}
        # Cap hit while a tool was pending: one final no-tools LLM call.
        wrap_messages = state["messages"] + [
            {
                "role": "system",
                "content": (
                    "No more tool calls allowed. Answer the user now "
                    "using only the information already gathered."
                ),
            }
        ]
        wrap = llm_fn(wrap_messages)
        if isinstance(wrap, dict) and wrap.get("type") == "response":
            return {"final_response": wrap["text"]}
        return {"final_response": "I couldn't complete that lookup."}

    # State schema is the BrainState TypedDict (messages/tool_calls append,
    # everything else overwrites).
    g = StateGraph(BrainState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_node("respond", respond)
    g.set_entry_point("think")
    g.add_conditional_edges("think", route, {"act": "act", "respond": "respond"})
    g.add_edge("act", "think")
    g.add_edge("respond", END)
    return g.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


class AgenticBrain:
    """Runs one user turn through the think/act/observe loop.

    Usage:
        brain = AgenticBrain(llm_fn=my_groq_caller)
        text = await brain.arun("What's the weather in Dallas?")
    """

    def __init__(
        self,
        llm_fn: Callable[[list[dict]], dict],
        tools: dict[str, Callable[..., str]] | None = None,
        max_rounds: int = MAX_ROUNDS,
    ):
        self._graph = build_graph(llm_fn, tools=tools, max_rounds=max_rounds)

    async def arun(self, user_text: str) -> str:
        """Run the graph for one user turn; return the final response text."""
        import asyncio

        result = await asyncio.to_thread(self._graph.invoke, _initial_state(user_text))
        return result.get("final_response") or ""

    def run(self, user_text: str) -> str:
        """Sync wrapper (for tests / non-async callers)."""
        import asyncio

        return asyncio.run(self.arun(user_text))


# ---------------------------------------------------------------------------
# Pipecat wrapper (optional — only defined when pipecat is installed)
# ---------------------------------------------------------------------------

try:
    from pipecat.frames.frames import TextFrame, TranscriptionFrame
    from pipecat.processors.frame_processor import FrameProcessor

    _PIPECAT_AVAILABLE = True
except ImportError:  # pragma: no cover - test env may lack pipecat
    _PIPECAT_AVAILABLE = False


if _PIPECAT_AVAILABLE:

    class BrainProcessor(FrameProcessor):
        """Pipecat processor: user text in -> response text out.

        Accepts TextFrame (tests / direct use) and TranscriptionFrame (STT
        output in a real pipeline); emits the final response as a single
        TextFrame. Sentence streaming is a later step.
        """

        def __init__(self, brain: "AgenticBrain", **kwargs):
            super().__init__(**kwargs)
            self._brain = brain

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            # Only handle user text; pass everything else through.
            if isinstance(frame, (TextFrame, TranscriptionFrame)):
                try:
                    response = await self._brain.arun(frame.text)
                except Exception as e:
                    logger.exception("AgenticBrain failed")
                    response = "Sorry, I ran into an error."
                await self.push_frame(TextFrame(text=response))
            else:
                await self.push_frame(frame, direction)
