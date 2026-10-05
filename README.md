# Speech-to-Speech Voice Lab

A real-time voice pipeline on Hugging Face's
[`speech-to-speech`](https://github.com/huggingface/speech-to-speech),
deployed as **one hybrid architecture**: a free always-on CPU box that routes
STT/TTS to pay-per-use serverless GPUs, with local CPU fallback.

```
Mac ──ws──▶ Oracle Always-Free (24/7, $0): full CPU pipeline + GPU routers
                  │   race GPU vs CPU per turn ──first finisher wins──▶
                  └─ RunPod serverless GPUs (pay-per-second, $0 when idle)
```

**Setup: [`SETUP.md`](SETUP.md)** — the single end-to-end guide
(Oracle box → GPU legs → Mac client → Colab alternative).

## Layout

| Dir | What |
|---|---|
| `oracle/` | Everything that runs on the box: setup script, systemd units, tool server, GPU router plugin |
| `serverless-gpu/runpod/` | The two RunPod endpoint builds (Dockerfiles + handlers) |
| `colab/` | Colab T4 notebook (alternative path, no deploy) |
| `mac/` | Thin client: `slow_tools.py` forwarder |

## Architecture notes

- The LLM is remote in every path (`openai/gpt-oss-20b:groq` via the HF
  Inference Providers router) — no local GPU needed for the brain.
- Tool *definitions* travel Mac → server → model; tool *execution* runs on the
  server (`tool_server.py`, port 8766). Async pattern: tools never block the
  turn — long jobs run in the background, results land on follow-up turns.
- Per-stage latency is measurable from server logs — the pipeline is built
  to be debugged stage by stage.

## Secrets

No secrets are stored in this repo.

| Secret | Where it's read |
|---|---|
| `HF_TOKEN` | Colab Secrets (notebook); `/etc/s2s/env` on Oracle |
| `RUNPOD_API_KEY`, `RUNPOD_STT_ENDPOINT_ID`, `RUNPOD_TTS_ENDPOINT_ID` | `/etc/s2s/env` on Oracle |
| `NGROK_TOKEN`, `NGROK_DOMAIN` | Colab Secrets (notebook, Colab path only) |
