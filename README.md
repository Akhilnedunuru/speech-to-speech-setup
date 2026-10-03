# Speech-to-Speech Voice Lab

A working, debuggable real-time voice pipeline built on Hugging Face's
[`speech-to-speech`](https://github.com/huggingface/speech-to-speech) repo:

```
mic → Silero VAD → Smart Turn → STT → LLM (Groq) → TTS → speaker
                                          ↕ tools (claim lookup)
```

## Layout

| Dir | What |
|---|---|
| `colab/` | Google Colab notebook — GPU path (T4): Parakeet STT + Qwen3-TTS, ngrok tunnels (stable speech URL + tool API) |
| `oracle/` | Oracle Always-Free Ampere VM — free 24/7 CPU path: faster-whisper + Kokoro, systemd services, direct public IP (no tunnel) |
| `mac/` | Thin Mac client: mic/speaker `talk` commands + tool module (forwards tool calls to the server) |

## Architecture notes

- The LLM is remote in both paths (`openai/gpt-oss-20b:groq` via the HF
  Inference Providers router) — no local GPU needed for the brain.
- Tool *definitions* travel Mac → server → model; tool *execution* runs on the
  server (`tool_server.py`, port 8766). Async pattern: tools never block the
  turn — long jobs run in the background, results land on follow-up turns.
- Per-stage latency is measurable from server logs
  (`Smart Turn:` / `final STT done` / `TTFA` / `RTF` lines) — the pipeline is
  built to be debugged stage by stage.

## Secrets

No secrets are stored in this repo. The Colab notebook reads `HF_TOKEN` /
`NGROK_TOKEN` / `NGROK_DOMAIN` from Colab Secrets; the Oracle box reads
`HF_TOKEN` from `~/.s2s_env`.
