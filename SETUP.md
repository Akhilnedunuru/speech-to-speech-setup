# Setup — hybrid voice pipeline (the only setup doc)

End-to-end: Oracle Always-Free box (Part A) → RunPod GPU legs (Part B) →
Mac client. Colab alternative at the end. Follow it top to bottom.

```
Mac ──ws──▶ Oracle Always-Free (24/7, $0): full CPU pipeline + GPU routers
                  │   try GPU first ──cold/timeout/error──▶ CPU fallback
                  └─ RunPod serverless GPUs (pay-per-second, $0 when idle)
```

## 0. What you're building

```
mic → Silero VAD → Smart Turn → STT → LLM (Groq, remote) → TTS → speaker
                                          ↕ tools (claim lookup)
```

| Stage | GPU leg (Part B) | CPU fallback (Part A) |
|---|---|---|
| VAD / Smart Turn | — | CPU (always local) |
| STT | Parakeet TDT 0.6B | faster-whisper `small.en` int8 |
| LLM | — | `openai/gpt-oss-20b:groq` via HF router (always remote) |
| TTS | Qwen3-TTS 1.7B | Kokoro 82M (`af_heart`) |
| Tools | — | `tool_server.py` on :8766 |

The router races the GPU endpoint against local CPU every turn — first
finisher wins. Cold GPU (slow) → CPU answers with zero added latency while
the GPU request keeps warming the worker in the background, so the next
turns hit a warm GPU. Warm GPU (fast) → GPU wins on quality. Only real GPU
errors (not slowness) count toward the circuit breaker, which skips a dead
endpoint for 5 minutes.

**Cost:** Oracle $0 forever. RunPod ~$0.25–0.35/GPU-hour, billed per second —
roughly $10–25/month at ~1 hr/day of talking, $0 when idle.

**Secrets** (none stored in this repo): `HF_TOKEN` → `/etc/s2s/env` on Oracle
(Colab Secrets for the notebook path); `RUNPOD_API_KEY`,
`RUNPOD_STT_ENDPOINT_ID`, `RUNPOD_TTS_ENDPOINT_ID` → `/etc/s2s/env`;
`NGROK_TOKEN`, `NGROK_DOMAIN` → Colab Secrets only.

---

## Part A — the always-on box (Oracle Always Free, ~45 min)

The 24/7 front door: a free Ampere VM (2 ARM cores, 12 GB RAM) running the
full pipeline on CPU. Get this talking first; Part B only upgrades STT/TTS.

### A1. Oracle account

1. Sign up at cloud.oracle.com (Always Free tier). A credit card is required
   **for verification only** — the Always Free resources are not charged.
2. Pick a home region near Texas. Ampere "out of capacity" is common — try
   every availability domain, retry at off-peak hours, or consider a smaller
   shape (2 OCPU / 12 GB also runs this pipeline) and resize later.

### A2. Create the VM

Compute → Instances → Create:
- Image: **Ubuntu 24.04**, shape **VM.Standard.A1.Flex** (Ampere ARM)
- OCPUs: **2**, Memory: **12 GB**, Boot volume: 100 GB (all within Always Free)
- Add your SSH public key (`~/.ssh/id_ed25519.pub` on your Mac)
- Note the **public IP** after it boots

### A3. Open ports

1. In the OCI console: VCN → Security List → add Ingress rules for
   **TCP 8765** and **TCP 8766** from `0.0.0.0/0`.
2. On the VM itself the setup script handles iptables (ports 8765/8766).

### A4. Install

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@<PUBLIC-IP>
# on the VM — clone the repo (always gets the latest code) and run setup:
git clone https://github.com/Akhilnedunuru/speech-to-speech-setup
cd speech-to-speech-setup/oracle
chmod +x setup_oracle.sh && ./setup_oracle.sh
```

### A5. Secrets + services

```bash
# on the VM — one secrets file for both services (Part B appends RUNPOD_* here)
sudo tee /etc/s2s/env > /dev/null <<'EOF'
HF_TOKEN=<your-hf-token>
EOF
sudo chmod 600 /etc/s2s/env

sudo cp ~/s2s.service ~/s2s-tools.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now s2s s2s-tools
systemctl status s2s            # should be active
journalctl -u s2s -f            # watch startup logs (~1-2 min first boot)
```

systemd restarts both services if they crash, and they come back up on VM reboot.
That's your 24/7.

### A6. Verify: talk to the CPU pipeline

Set up the Mac client once (§3), then point it at the box:

```bash
source ~/Desktop/s2s-client/bin/activate
export OPENAI_API_KEY=not-needed
export TOOL_API_URL=http://<PUBLIC-IP>:8766
PYTHONPATH=$HOME/Desktop speech-to-speech talk \
  --url ws://<PUBLIC-IP>:8765/v1/realtime \
  --playback-buffer-ms 800 \
  --block-mic-during-playback \
  --tool-module slow_tools \
  --instructions "You are a helpful voice assistant for an insurance company. ..."
```

Say a sentence. It should answer in ~4–8s/turn (CPU STT/TTS). If it does,
Part A is done — continue to Part B whenever you want GPU speed.

---

## Part B — serverless GPU legs (~1.5 h, mostly waiting)

**Prerequisite: Part A talks.** Builds two RunPod endpoints and flips the
box's STT/TTS to `runpod-routed`. To go back to pure CPU any time:
`sudo systemctl stop s2s-routed && sudo systemctl start s2s` — nothing else changes.

### B1. RunPod account (10 min)

1. Sign up at [runpod.io](https://www.runpod.io), add a small amount of credit
   (there's a minimum top-up — $10ish; serverless deducts per-second from balance).
2. Create an API key: **Settings → API Keys → Create**. Save as `RUNPOD_API_KEY`.
3. Create a Docker Hub account (free) — RunPod pulls your images from a registry.

### B2. Build & push the two images (30–60 min, mostly waiting)

RunPod GPUs are x86_64, your Mac is ARM, so build with `buildx` for
`linux/amd64`. Needs Docker Desktop running.

```bash
cd <repo>/serverless-gpu
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

### B3. Create the two serverless endpoints (10 min)

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

### B4. Smoke-test both endpoints (5 min)

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

### B5. Install the router on the Oracle box (15 min)

SSH into the VM (Part A is running there), then:

```bash
# one extra dep, into the existing venv
~/s2s/bin/pip install -r ~/speech-to-speech-setup/oracle/requirements-router.txt

# router files (in the repo's oracle/ dir — copy that dir to the VM, or clone the repo there)
mkdir -p ~/s2s-router
cp ~/speech-to-speech-setup/oracle/router_plugin.py ~/speech-to-speech-setup/oracle/serve_routed.py ~/s2s-router/

# append the RunPod secrets to the same env file Part A created
sudo tee -a /etc/s2s/env > /dev/null <<'EOF'
RUNPOD_API_KEY=...
RUNPOD_STT_ENDPOINT_ID=...
RUNPOD_TTS_ENDPOINT_ID=...
EOF
```

Env knobs (optional, all have sane defaults):
`GPU_ROUTE_TIMEOUT_S=8` (give up on GPU this turn after 8 s),
`GPU_ROUTE_MAX_FAILURES=3`, `GPU_ROUTE_COOLDOWN_S=300`.

Dry-run in the foreground first (stop the Part A service so the port is free):

```bash
sudo systemctl stop s2s
export $(sudo cat /etc/s2s/env | xargs)
~/s2s/bin/python ~/s2s-router/serve_routed.py \
  --host 0.0.0.0 --port 8765 \
  --stt runpod-routed --tts runpod-routed \
  --faster_whisper_stt_model_name small.en \
  --faster_whisper_stt_compute_type int8 \
  --kokoro_voice af_heart \
  --kokoro_lang_code a \
  --llm_backend responses-api \
  --model_name "openai/gpt-oss-20b:groq" \
  --responses_api_base_url "https://router.huggingface.co/v1" \
  --responses_api_api_key "$HF_TOKEN" \
  --responses_api_stream \
  --responses_api_reasoning_effort low \
  --no_enable_live_transcription
```

> `--responses_api_reasoning_effort low` is required: the HF router rejects
> the default `none` with a 400 error during the LLM warmup.
> `--no_enable_live_transcription` disables mid-speech progressive STT passes
> (each burned 1–2 s of CPU and stalled the final transcription).
```

What the router does per turn:
- **Cold:** GPU and CPU race; first finisher wins. GPU timeout (8 s) falls
  back to CPU with no added latency; the GPU request keeps warming the
  worker in the background. First STT race also fires a background warmup
  for the TTS endpoint and vice versa.
- **Warm:** after a fast GPU win, later turns skip the CPU leg entirely
  (`warm shortcut`). Idle > 240 s marks the endpoint cold again (RunPod
  scales to zero at 300 s).
- **TTS streaming:** warm replies are split into ~80-char clauses, each
  sent to the GPU separately; audio plays as clauses arrive (first sound
  ~1–2 s). If the GPU fails mid-stream, only the unspoken remainder falls
  back to CPU.
- **Safety:** empty GPU audio raises so CPU backs it up; 3 real GPU errors
  trip the breaker (5 min cooldown). Timeouts never trip it.

You should see `Registered backends: --stt runpod-routed, --tts runpod-routed`
in the log, then the normal server startup.

### B6. Switch systemd to routed (2 min)

```bash
sudo cp ~/speech-to-speech-setup/oracle/s2s-routed.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl stop s2s                     # pure-CPU service from Part A
sudo systemctl enable --now s2s-routed
sudo systemctl status s2s-routed            # check it's active
tail -f ~/s2s-router/server.log
```

The tool server (`tool_server.py` on :8766 + `s2s-tools.service`) is unchanged
from Part A — keep it running as before.

### B7. Verification checklist

1. **GPU path works:** talk one sentence → server.log shows
   `STT won by gpu` / `TTS won by gpu` with timings.
2. **Cold start is graceful, not invisible:** wait 10+ min idle (workers
   scaled to zero), talk → first 3–5 turns log `won by cpu` (GPU still
   booting; the race falls back with no added latency) and `GPU leg timed
   out after 8s (cold worker?)`. STT usually warms by turn 3–4, TTS by
   turn 5–6 (its cold start is 40–60 s: GPU allocation + large image pull).
   You hear no error, just slower turns.
3. **Warm shortcut:** once a GPU leg wins quickly, later turns log
   `won by gpu (warm shortcut)` — the CPU leg is skipped entirely, so the
   2-core box isn't burning cycles on discarded work.
4. **Streaming TTS:** on warm turns the log shows
   `tts: GPU streamed sentence 1/2 (... chars) -> ...s audio` — replies are
   split into ~80-char clauses, first sound lands ~1–2 s after the LLM
   finishes instead of waiting for the whole reply.
5. **Breaker works:** stop both endpoints (or break the API key) → after
   3 failed turns, no more GPU attempts for 5 min; everything still answers.
   Timeouts don't trip the breaker (cold workers aren't errors).
6. **Bill check:** RunPod console → Billing — after a day of tinkering you
   should see minutes of GPU time, not hours.

Expected warm-turn budget: ~1 s VAD→STT handoff + 0.7 s STT + 0.5 s LLM +
~2 s first TTS clause + 0.8 s Mac playback buffer ≈ 4–5 s total.

---

## 3. Mac client (one-time setup)

The Mac is the thin client: microphone + speaker + tool definitions. All heavy
work runs on the server.

```bash
python3.11 -m venv ~/Desktop/s2s-client
source ~/Desktop/s2s-client/bin/activate
pip install speech-to-speech
# macOS SSL fix for python.org Python (if needed):
/Applications/Python\ 3.11/Install\ Certificates.command
```

Save `slow_tools.py` next to the venv (e.g. `~/Desktop/`).

Talk against the Oracle box (works with Part A or Part B — same command):

```bash
source ~/Desktop/s2s-client/bin/activate
export OPENAI_API_KEY=not-needed
export TOOL_API_URL=http://<PUBLIC-IP>:8766
PYTHONPATH=$HOME/Desktop speech-to-speech talk \
  --url ws://<PUBLIC-IP>:8765/v1/realtime \
  --playback-buffer-ms 800 \
  --block-mic-during-playback \
  --tool-module slow_tools \
  --instructions "You are a helpful voice assistant for an insurance company. When the user asks about a claim: speak 'Let me start that lookup for you.', call lookup_claim_status, then tell the user the search is running in the background (about 10 seconds) and they can ask about anything else meanwhile. When they follow up, call get_claim_result with the search_id. Never invent claim details."
```

Notes:
- `OPENAI_API_KEY=not-needed` — the SDK requires a non-empty key; our server ignores it.
- `PYTHONPATH` must point at the directory containing `slow_tools.py`
  (console scripts don't add the cwd to the import path).
- `--block-mic-during-playback` prevents the mic hearing the reply (kills barge-in;
  headphones are the better fix if you want barge-in).
- The URL must be the full Realtime endpoint ending in `/v1/realtime` —
  a bare `ws://<IP>:8765` fails with `ValueError: --url must end in /realtime`.
- Note `ws://` (not `wss://`) — no TLS on the raw IP, fine for personal dev.
  For `wss://`, point a free DuckDNS subdomain at the IP and put Caddy in front.

---

## 4. Alternative: Colab T4 (no deploy)

Same pipeline on a Colab GPU instead of the hybrid. **Use for:** zero-commitment
dev, experiments, model swaps. **Don't use for:** anything that must stay up —
Colab reclaims the VM (hours, always within ~24h).

Notebook: `colab/speech-to-speech-colab.ipynb`. GPU stages on the T4:
Parakeet TDT STT + Qwen3-TTS (same models as the hybrid's RunPod legs); LLM is
remote either way. Two ngrok tunnels: stable speech endpoint
(`wss://…/v1/realtime`) + a second tunnel for the tool API the Mac forwards to.
Secrets via Colab Secrets: `HF_TOKEN`, `NGROK_TOKEN`, `NGROK_DOMAIN`.

Talk via Colab:

```bash
source ~/Desktop/s2s-client/bin/activate
export OPENAI_API_KEY=not-needed
export TOOL_API_URL=<from Colab tunnel cell>
PYTHONPATH=$HOME/Desktop speech-to-speech talk \
  --url wss://<your-ngrok-domain>/v1/realtime \
  --playback-buffer-ms 800 \
  --block-mic-during-playback \
  --tool-module slow_tools \
  --instructions "You are a helpful voice assistant for an insurance company. ..."
```

---

## 5. How tool calling works here

The model never executes tools — it emits a structured function call, the app
runs it, and the result goes back into the conversation. Tool *definitions*
travel Mac → server → model; tool *execution* runs on the server
(`tool_server.py`, port 8766). Async pattern: tools never block the turn —
long jobs run in the background, results land on follow-up turns.
(`slow_tools.py` on the Mac is a thin forwarder to `TOOL_API_URL`.)

## 6. Hacking on the server

Just SSH in. The pip install is a wheel; to hack the source:

```bash
git clone https://github.com/huggingface/speech-to-speech ~/src
source ~/s2s/bin/activate
pip install -e "$HOME/src/[faster-whisper,kokoro]"   # editable install
sudo systemctl restart s2s        # or s2s-routed, whichever is active
```

Per-stage latency is measurable from server logs (`Smart Turn:` /
`final STT done` / `TTFA` / `RTF` lines, plus `via RunPod GPU` / `CPU fallback`
from the router) — the pipeline is built to be debugged stage by stage.

## 7. Troubleshooting

- `journalctl -u s2s -n 50` / `-u s2s-routed` / `-u s2s-tools` — server logs
- `curl localhost:8766/run-tool -X POST -d '{"name":"get_claim_result","arguments":{"search_id":"x"}}'`
  → `{"status":"error",...}` means the tool API is reachable
- `ModuleNotFoundError: faster_whisper` → you installed `speech-to-speech`
  without the `[faster-whisper,kokoro]` extras
- `Unsupported backend 'runpod-routed'` → `serve_routed.py` wasn't used
  (it must run instead of `speech-to-speech serve`), or the registration
  import failed — check the log's first lines
- Every turn falls back, breaker never trips → likely a bad endpoint ID or
  image pull failure — check the endpoint's own logs in the RunPod console
- TTS sounds wrong/robotic → check `/tmp/tts_test.wav` from B4 first to
  isolate endpoint vs router
- `curl` to runsync hangs forever on first call → normal cold start
  (up to ~2 min with model download). Don't Ctrl-C; let it complete once
- RunPod console test shows Failed in ~275 ms → the console's default
  payload uses `"prompt"`; our TTS handler expects `"text"`. Change the key
  and re-run. (Pipeline traffic via `runsync` is unaffected.)
- TTS delay time 40–60 s in the console → normal cold start (GPU allocation
  + image pull). STT is faster (smaller image). While testing, bump the
  endpoint's Idle Timeout to 900–1800 s so workers stay warm between rounds;
  set it back to 300 s when done.
- `GPU leg timed out after 8s (cold worker?)` on every turn for 5+ turns →
  check the endpoint's Requests tab: long *delay* time = waiting on GPU
  capacity; long *execution* time = slow worker. Also verify the GPU types
  (RTX 4000 Ada / A4000 / A4500 / 2000 Ada are all fine — more types = better
  availability).
- Turn shows `audio=0.00s` right after a tool call → metric misattribution,
  not silence: the spoken reply's audio gets counted on the tool-call
  response line. You heard it.
- No `faster_whisper - Processing audio` lines during speech → expected:
  `--no_enable_live_transcription` disables the 0.5 s progressive passes
  (they were stalling the STT thread ~2 s/turn). Final transcriptions still
  appear.

## 8. Cost guardrails (set once, sleep well)

- Max workers = 1 on both endpoints (set in B3).
- Idle timeout 300 s → $0 between conversations.
- RunPod has spend-limit / alert settings — set a monthly cap that pages you.
- If a runaway ever happens: set Max workers to 0 on both endpoints to
  instantly stop all GPU spend; the Oracle CPU pipeline keeps working.

## Appendix: file map

| Dir | What |
|---|---|
| `oracle/` | Everything that runs on the box: setup script, systemd units, tool server, GPU router plugin |
| `serverless-gpu/runpod/` | The two RunPod endpoint builds (Dockerfiles + handlers) |
| `colab/` | Colab T4 notebook (alternative path) |
| `mac/` | Thin client: `slow_tools.py` forwarder |
