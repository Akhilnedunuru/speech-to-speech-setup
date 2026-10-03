# ~/Desktop/slow_tools.py — THIN CLIENT. Menu unchanged; execution happens on Colab/Oracle.
# Requires: export TOOL_API_URL=https://...  (printed by the Colab tunnel cell,
#           or http://<ORACLE-IP>:8766 on the Oracle box)
import asyncio
import datetime
import json
import os
import urllib.request

TOOL_API_URL = os.environ.get("TOOL_API_URL", "").rstrip("/")

TOOLS = [
    {
        "type": "function",
        "name": "lookup_claim_status",
        "description": (
            "Start a claim-status lookup. Returns INSTANTLY with a search_id and "
            "status 'pending' — the backend takes ~10s. Tell the user the search is "
            "running in the background and they can ask about anything else while "
            "they wait, or say 'is it ready' in a few seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {"claim_number": {"type": "string"}},
            "required": ["claim_number"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_claim_result",
        "description": (
            "Check a running claim search. Returns the result if finished, or "
            "'still_pending'. If still pending, briefly tell the user it's still running."
        ),
        "parameters": {
            "type": "object",
            "properties": {"search_id": {"type": "string"}},
            "required": ["search_id"],
            "additionalProperties": False,
        },
    },
]

CREATE_RESPONSE = True


def _post_to_server(name, arguments):
    req = urllib.request.Request(
        TOOL_API_URL + "/run-tool",
        data=json.dumps({"name": name, "arguments": arguments}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


async def execute_tool(name, arguments):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[tool] {name} -> server at {ts}", flush=True)
    if not TOOL_API_URL:
        raise RuntimeError("TOOL_API_URL is not set — export it first")
    result = await asyncio.to_thread(_post_to_server, name, arguments)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[tool] {name} <- server at {ts}: status={result.get('status')}", flush=True)
    return result
