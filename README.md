# Speech-to-Speech Voice Lab

A real-time voice pipeline on Hugging Face's
[`speech-to-speech`](https://github.com/huggingface/speech-to-speech),
deployed as **one hybrid architecture**: a free always-on CPU box that routes
STT/TTS to pay-per-use serverless GPUs, with local CPU fallback.

```
Mac ──ws──▶ Oracle Always-Free (24/7, $0): full CPU pipeline + GPU routers
                  │   try GPU first ──cold/timeout/error──▶ CPU fallback
                  └─ RunPod serverless GPUs (pay-per-second, $0 when idle)
```

## The pipeline

```
mic → Silero VAD → Smart Turn → STT → LLM (Groq, remote) → TTS → speaker
                                          ↕ tools (claim lookup)
```

| Stage | GPU leg (RunPod serverless) | CPU fallback (Oracle) |
|---|---|---|
| VAD / Smart Turn | — | CPU (always local) |
| STT | Parakeet TDT 0.6B | faster-whisper `small.en` int8 |
| LLM | — | `openai/gpt-oss-20b:groq` via HF router (always remote) |
| TTS | Qwen3-TTS 1.7B | Kokoro 82M (`af_heart`) |
| Tools | — | `tool_server.py` on :8766 |

The router tries the GPU endpoint per turn with a short timeout. Cold start,
timeout, or error → the turn is answered from CPU with zero added latency, and
the GPU worker finishes warming for the next turn. A circuit breaker stops
trying a dead endpoint for 5 minutes.

**Cost:** Oracle $0 forever. RunPod ~$0.25–0.35/GPU-hour, billed per second —
roughly $10–25/month at ~1 hr/day of talking, $0 when idle.

## Layout

| Dir | What |
|---|---|
| `oracle/` | **Part A — the always-on box.** Setup script, systemd units, tool server, and the GPU router plugin. Start here. |
| `serverless-gpu/` | **Part B — the GPU legs.** RunPod Dockerfiles + handlers and the endpoint setup guide. Add after Part A works. |
| `colab/` | **Alternative — Colab T4.** Same pipeline on a Colab GPU (Parakeet + Qwen3-TTS), for dev/experiments. Ephemeral by design. |
| `mac/` | Thin client: mic/speaker `talk` commands + tool module (forwards tool calls to the server). |

## Deploy order

1. **Oracle box** (`oracle/README.md`) — VM, firewall, install, pure-CPU pipeline live 24/7.
2. **GPU legs** (`serverless-gpu/SETUP.md`) — build the two RunPod endpoints, flip the box to `runpod-routed`.
3. **Talk** — Mac client against the Oracle public IP.

`colab/` stays as the zero-commitment alternative: nothing to deploy, but the
runtime dies when Colab reclaims the VM.

## Architecture notes

- The LLM is remote in every path (`openai/gpt-oss-20b:groq` via the HF
  Inference Providers router) — no local GPU needed for the brain.
- Tool *definitions* travel Mac → server → model; tool *execution* runs on the
  server (`tool_server.py`, port 8766). Async pattern: tools never block the
  turn — long jobs run in the background, results land on follow-up turns.
- Per-stage latency is measurable from server logs
  (`Smart Turn:` / `final STT done` / `TTFA` / `RTF` lines, plus
  `via RunPod GPU` / `CPU fallback` from the router) — the pipeline is built
  to be debugged stage by stage.

## Secrets

No secrets are stored in this repo.

| Secret | Where it's read |
|---|---|
| `HF_TOKEN` | Colab Secrets (notebook); `/etc/s2s/env` on Oracle |
| `RUNPOD_API_KEY`, `RUNPOD_STT_ENDPOINT_ID`, `RUNPOD_TTS_ENDPOINT_ID` | `/etc/s2s/env` on Oracle |
| `NGROK_TOKEN`, `NGROK_DOMAIN` | Colab Secrets (notebook, Colab path only) |
