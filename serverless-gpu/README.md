# Part B — serverless GPU legs

This is step 2 of the hybrid architecture (the repo's main path).
Start with `oracle/` (Part A — the always-on box), then come back here.

- **`SETUP.md`** — the full guide: RunPod account → build/push the two images →
  create the endpoints → wire the Oracle box to `runpod-routed` → verify.
- **`runpod/stt/`** — Parakeet TDT 0.6B serverless endpoint (Dockerfile + handler).
- **`runpod/tts/`** — Qwen3-TTS 1.7B serverless endpoint (Dockerfile + handler).

The router files (`router_plugin.py`, `serve_routed.py`, `s2s-routed.service`)
live in the top-level `oracle/` dir — they execute on the box, so they live
with the box setup.

Cost: ~$0.25–0.35/GPU-hour, billed per second, $0 when idle.
