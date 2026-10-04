# Hybrid setup: Oracle 24/7 CPU + RunPod serverless GPU (pay-per-use)

```
Mac ──ws──▶ Oracle (free, 24/7): full CPU pipeline + smart routers
                  │
                  ├─ STT/TTS router: try serverless GPU ──cold/timeout──▶ local CPU fallback
                  │
                  └─ RunPod Serverless: Parakeet STT + Qwen3-TTS endpoints
                     → scale to zero after 5 min idle → $0 when you're not talking
```

**What you get:** the Oracle box from Phase 1 stays the always-on front door
(free forever). STT and TTS each try a RunPod GPU endpoint first (same models
as your Colab rig: Parakeet TDT 0.6B + Qwen3-TTS 1.7B). Cold start, timeout, or
error → the turn is answered by local CPU (faster-whisper / Kokoro) with zero
added latency, and the GPU worker finishes warming for the next turn. A circuit
breaker stops trying a dead endpoint for 5 minutes.

**Cost:** ~$0.25–0.35/GPU-hour, billed per second, only while a worker is up.
At ~1 hr/day of talking ≈ **$10–25/month**. $0 when idle.

**Files in this folder:**
- `runpod/stt/` — Dockerfile + handler (Parakeet TDT, via `nano-parakeet`, same as Colab)
- `runpod/tts/` — Dockerfile + handler (Qwen3-TTS ggml, same as Colab)
- `oracle/router_plugin.py` — the two router backends + circuit breaker
- `oracle/serve_routed.py` — launcher that registers them, then runs stock `serve`
- `oracle/s2s-routed.service` — systemd unit
- `oracle/requirements-router.txt` — one extra dep (`requests`)

---

## Part A — RunPod account (10 min)

1. Sign up at [runpod.io](https://www.runpod.io), add a small amount of credit
   (there's a minimum top-up — $10ish; serverless deducts per-second from balance).
2. Create an API key: **Settings → API Keys → Create**. Save as `RUNPOD_API_KEY`.
3. Create a Docker Hub account (free) — RunPod pulls your images from a registry.

## Part B — Build & push the two images (30–60 min, mostly waiting)

RunPod GPUs are x86_64, your Mac is ARM, so build with `buildx` for
`linux/amd64`. Needs Docker Desktop running.

```bash
cd ~/workspace/your_files/serverless-gpu   # or wherever you cloned the repo
export DH=<your-dockerhub-username>

# STT image
docker buildx build --platform linux/amd64 \
  -t $DH/s2s-parakeet-stt:latest --push runpod/stt

# TTS image
docker buildx build --platform linux/amd64 \
  -t $DH/s2s-qwen3-tts:latest --push runpod/tts
```

> Emulated amd64 builds on Apple Silicon are slow. If a build stalls, it's
> normal — the TTS image compiles the ggml backend. Go make coffee.

## Part C — Create the two serverless endpoints (10 min)

Do this twice (once per image). In the RunPod console: **Serverless → New Endpoint**.

| Setting | STT endpoint | TTS endpoint |
|---|---|---|
| Name | `s2s-parakeet-stt` | `s2s-qwen3-tts` |
| Container image | `$DH/s2s-parakeet-stt:latest` | `$DH/s2s-qwen3-tts:latest` |
| GPU type | cheapest with ≥16 GB VRAM (e.g. RTX A4000 / 4000 Ada) | same |
| Min / Max workers | 0 / 1 | 0 / 1 |
| Idle timeout | 300 s (stays warm through a conversation, then $0) | 300 s |
| Execution timeout | 120 s | 180 s (long sentences) |
| Env vars | `HF_TOKEN=<your token>` | `HF_TOKEN=<your token>` |

Copy each endpoint's **Endpoint ID** (the hex string in its URL).

**Why max workers = 1:** you're the only user; one worker serves turns
sequentially, and it caps the bill.

## Part D — Smoke-test both endpoints (5 min)

```bash
export RUNPOD_API_KEY=... STT_ID=... TTS_ID=...

# STT: send 2s of tone (expect COMPLETED; text will be empty — that proves the path)
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 /tmp/tone.wav
curl -s https://api.runpod.ai/v2/$STT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "{\"input\": {\"audio\": \"$(base64 -w0 /tmp/tone.wav)\"}}" | head -c 300; echo

# TTS: expect COMPLETED + a wav file
curl -s https://api.runpod.ai/v2/$TTS_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"text": "Hello, this is a test of the GPU voice pipeline."}}' \
  | python3 -c "
import json, sys, base64
d = json.load(sys.stdin)
print('status:', d.get('status'))
open('/tmp/tts_test.wav','wb').write(base64.b64decode(d['output']['audio']))"
afplay /tmp/tts_test.wav   # should sound like the Colab voice (Aiden)
```

> First call takes 60–120 s (cold start: pod boot + model download + load).
> `curl` has no timeout here on purpose — let it finish. Later calls: ~1–3 s.

## Part E — Oracle: install the router (15 min)

SSH into the Oracle VM (Phase 1 box), then:

```bash
mkdir -p ~/s2s-router && cd ~/s2s-router
python3 -m venv venv && source venv/bin/activate
pip install "speech-to-speech[faster-whisper,kokoro]" requests

# copy these three files from the repo: router_plugin.py, serve_routed.py
# (scp, or git clone your speech-to-speech-setup repo)
```

Create the secrets file (never in git):

```bash
sudo mkdir -p /etc/s2s && sudo chmod 700 /etc/s2s
sudo tee /etc/s2s/env > /dev/null <<'EOF'
HF_TOKEN=...
RUNPOD_API_KEY=...
RUNPOD_STT_ENDPOINT_ID=...
RUNPOD_TTS_ENDPOINT_ID=...
EOF
sudo chmod 600 /etc/s2s/env
```

Env knobs (optional, all have sane defaults):
`GPU_ROUTE_TIMEOUT_S=8` (give up on GPU this turn after 8 s),
`GPU_ROUTE_MAX_FAILURES=3`, `GPU_ROUTE_COOLDOWN_S=300`.

Dry-run in the foreground first:

```bash
set -a; source /etc/s2s/env; set +a   # not needed if you export them yourself
./venv/bin/python serve_routed.py \
  --host 0.0.0.0 --port 8765 \
  --stt runpod-routed --tts runpod-routed \
  --faster_whisper_stt_model_name small.en \
  --kokoro_voice af_heart \
  --llm_backend responses-api \
  --model_name "openai/gpt-oss-20b:groq" \
  --responses_api_base_url "https://router.huggingface.co/v1" \
  --responses_api_api_key "$HF_TOKEN" \
  --responses_api_stream
```

You should see `Registered backends: --stt runpod-routed, --tts runpod-routed`
in the log, then the normal server startup.

## Part F — systemd (2 min)

```bash
sudo cp s2s-routed.service /etc/systemd/system/
sudo sed -i "s|/home/ubuntu|/home/$USER|g" /etc/systemd/system/s2s-routed.service
sudo systemctl daemon-reload
sudo systemctl enable --now s2s-routed
sudo systemctl status s2s-routed   # check it's active
tail -f ~/s2s-router/server.log
```

The tool server (`tool_server.py` on :8766 + `s2s-tools.service`) is unchanged
from Phase 1 — keep it running as before.

## Part G — Mac client (unchanged, new address)

Same as Phase 1, pointed at the Oracle public IP:

```bash
speech-to-speech talk --url ws://<ORACLE-PUBLIC-IP>:8765
```

## Verification checklist

1. **GPU path works:** talk one sentence → server.log shows
   `STT via RunPod GPU` / `TTS via RunPod GPU` with timings.
2. **Cold start is invisible:** wait 10+ min idle (worker scaled to zero),
   talk → first turn logs `RunPod STT failed (...)` + `CPU fallback`
   (that's the 8 s timeout firing), second turn goes GPU. You hear no error.
3. **Breaker works:** stop both endpoints (or break the API key) → after
   3 failed turns, no more GPU attempts for 5 min; everything still answers.
4. **Bill check:** RunPod console → Billing — after a day of tinkering you
   should see minutes of GPU time, not hours.

## Troubleshooting

- `ModuleNotFoundError: faster_whisper` → you installed `speech-to-speech`
  without the `[faster-whisper,kokoro]` extras.
- `Unsupported backend 'runpod-routed'` → `serve_routed.py` wasn't used
  (it must run instead of `speech-to-speech serve`), or the registration
  import failed — check the log's first lines.
- Every turn falls back, breaker never trips → likely a bad endpoint ID or
  image pull failure — check the endpoint's own logs in the RunPod console.
- TTS sounds wrong/robotic → the endpoint returned audio; check
  `/tmp/tts_test.wav` from Part D first to isolate endpoint vs router.
- `curl` to runsync hangs forever on first call → normal cold start
  (up to ~2 min with model download). Don't Ctrl-C; let it complete once.

## Cost guardrails (set once, sleep well)

- Max workers = 1 on both endpoints (set in Part C).
- Idle timeout 300 s → $0 between conversations.
- RunPod has spend-limit / alert settings — set a monthly cap that pages you.
- If a runaway ever happens: set Max workers to 0 on both endpoints to
  instantly stop all GPU spend; the Oracle CPU pipeline keeps working.

## Phase-2 note (voice cloning)

The TTS endpoint is already structured for it: add an optional
`ref_audio_b64` + `ref_text` to the input and call
`model.generate_voice_clone_streaming(...)` instead of
`generate_custom_voice_streaming(...)`. The router passes `text`/`language`
through today, so the only change is the endpoint handler plus one new input
field. The Oracle CPU path (Kokoro) can't clone — routed turns will simply
prefer the GPU endpoint whenever a cloned voice is requested.
