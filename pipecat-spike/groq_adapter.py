"""Groq adapter for the LangGraph brain (Phase 3, step 5).

Translates between AgenticBrain's llm_fn contract and Groq's chat API:

    llm_fn(messages) -> {"type": "tool", "name": str, "args": dict}
                     | {"type": "response", "text": str}

Tool-call convention: the system prompt instructs the model to reply with a
lone JSON object {"tool": "<name>", "args": {...}} when it wants a tool.
Anything else is treated as the final spoken response.

No pipecat dependency on purpose: this module is unit-testable anywhere.
"""

import json
import logging
import os
import re

logger = logging.getLogger("pipecat-groq-adapter")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = (
    "You are a concise voice assistant. Keep every reply short and "
    "conversational, suitable for speech. You have these tools: "
    "get_weather(city: string). "
    'If you need a tool, reply with ONLY a JSON object like '
    '{"tool": "get_weather", "args": {"city": "Dallas"}}. '
    "Otherwise reply with plain text only, no JSON."
)

# Matches a JSON object possibly wrapped in ```json fences.
_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_groq_client = None


def _get_client():
    global _groq_client
    if _groq_client is None:
        from openai import OpenAI

        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        _groq_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return _groq_client


def to_chat_messages(messages: list) -> list:
    """Convert brain messages to provider-safe chat messages.

    The brain uses {"role": "tool", ...} entries; most chat APIs expect
    tool results to arrive as user/assistant turns, so tool results are
    rewritten as user messages. Never mutates the input.
    """
    converted = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": f"[tool {m.get('name')}] {content}",
                }
            )
        else:
            converted.append({"role": role, "content": content})
    return converted


def _extract_tool_call(text: str):
    """Return (name, args) if text is a lone tool-call JSON object, else None."""
    candidate = text.strip()
    fence = _JSON_RE.search(candidate)
    if fence:
        candidate = fence.group(1)
    if not candidate.startswith("{"):
        return None
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
        args = parsed.get("args")
        return parsed["tool"], args if isinstance(args, dict) else {}
    return None


def groq_llm_fn(messages: list) -> dict:
    """AgenticBrain llm_fn backed by Groq chat completions."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
        + to_chat_messages(messages),
        temperature=0.7,
        max_tokens=300,
    )
    text = (resp.choices[0].message.content or "").strip()

    tool_call = _extract_tool_call(text)
    if tool_call:
        name, args = tool_call
        logger.info("Brain requested tool: %s(%s)", name, args)
        return {"type": "tool", "name": name, "args": args}
    return {"type": "response", "text": text}
