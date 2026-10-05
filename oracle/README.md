# Part A — the always-on box (Oracle Always Free)

This is the 24/7 front door of the hybrid architecture: a free Ampere VM
(4 ARM cores, 24 GB RAM) running the full voice pipeline on CPU. No free GPU
stays up 24/7 — Colab/Kaggle are ephemeral by design — so the two GPU-hungry
stages get CPU-friendly substitutes here:

| Stage | This box (CPU) | GPU leg (Part B) |
|---|---|---|
| VAD / Smart Turn | CPU (unchanged) | — |
| STT | faster-whisper `small.en` int8 | Parakeet TDT 0.6B on RunPod |
| LLM | Groq `openai/gpt-oss-20b` via HF router (unchanged, still free) | — |
| TTS | Kokoro 82M (`af_heart`) | Qwen3-TTS 1.7B on RunPod |
| Tools | `tool_server.py` on :8766 (unchanged) | — |
| Public URL | none needed — the VM has a public IP | — |

Get this part working first (pure CPU, ~4–8s/turn). Part B
(`../serverless-gpu/SETUP.md`) then adds the serverless GPU legs and flips
STT/TTS to `runpod-routed` — same box, same Mac client, nothing else changes.

## Step 1 — Oracle account

1. Sign up at cloud.oracle.com (Always Free tier). A credit card is required
   **for verification only** — the Always Free resources are not charged.
2. Pick a home region near Texas. Ampere "out of capacity" is common — try
   every availability domain, retry at off-peak hours, or consider a smaller
   shape (2 OCPU / 12 GB also runs this pipeline) and resize later.

## Step 2 — Create the VM

Compute → Instances → Create:
- Image: **Ubuntu 24.04**, shape **VM.Standard.A1.Flex** (Ampere ARM)
- OCPUs: **4**, Memory: **24 GB**, Boot volume: 100 GB (all within Always Free)
- Add your SSH public key (`~/.ssh/id_ed25519.pub` on your Mac)
- Note the **public IP** after it boots

## Step 3 — Open ports

1. In the OCI console: VCN → Security List → add Ingress rules for
   **TCP 8765** and **TCP 8766** from `0.0.0.0/0`.
2. On the VM itself the setup script handles iptables (ports 8765/8766).

## Step 4 — Install

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@<PUBLIC-IP>
# copy these files over (from your Mac):
#   setup_oracle.sh  tool_server.py  s2s.service  s2s-tools.service
scp -i ~/.ssh/id_ed25519 setup_oracle.sh tool_server.py s2s.service s2s-tools.service ubuntu@<PUBLIC-IP>:~/
# on the VM:
chmod +x ~/setup_oracle.sh && ~/setup_oracle.sh
```

## Step 5 — Secrets + services

```bash
# on the VM — one secrets file for both services (Part B adds RUNPOD_* here later)
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

## Step 6 — Mac client (no tunnel!)

```bash
cd ~/Desktop
source s2s-client/bin/activate
export TOOL_API_URL=http://<PUBLIC-IP>:8766
PYTHONPATH=$HOME/Desktop speech-to-speech talk \
  --url ws://<PUBLIC-IP>:8765 \
  --playback-buffer-ms 800 \
  --block-mic-during-playback \
  --tool-module slow_tools \
  --instructions "You are a helpful voice assistant for an insurance company. ..."
```

Note `ws://` (not `wss://`) — no TLS on the raw IP, fine for personal dev.
(If you later want `wss://`, point a free DuckDNS subdomain at the IP and put
Caddy in front for automatic HTTPS.)

## Part B — add the GPU legs (later)

Once the CPU pipeline talks:

1. Follow `../serverless-gpu/SETUP.md` Parts A–D (RunPod account, build/push
   the two images, create the endpoints, smoke-test).
2. On the VM:
   ```bash
   ~/s2s/bin/pip install -r ~/serverless-gpu/oracle/requirements-router.txt
   mkdir -p ~/s2s-router
   cp ~/serverless-gpu/oracle/router_plugin.py ~/serverless-gpu/oracle/serve_routed.py ~/s2s-router/
   sudo tee -a /etc/s2s/env > /dev/null <<'EOF'
   RUNPOD_API_KEY=<your-runpod-key>
   RUNPOD_STT_ENDPOINT_ID=<stt-endpoint-id>
   RUNPOD_TTS_ENDPOINT_ID=<tts-endpoint-id>
   EOF
   sudo cp ~/serverless-gpu/oracle/s2s-routed.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl stop s2s && sudo systemctl enable --now s2s-routed
   journalctl -u s2s-routed -f
   ```
   (Assumes you cloned this repo to `~/serverless-gpu` on the VM, or copied
   the `serverless-gpu/` dir over like the other files.)
3. Talk again — the log now shows `STT via RunPod GPU` / `TTS via RunPod GPU`,
   with `CPU fallback` on cold starts. To go back to pure CPU:
   `sudo systemctl stop s2s-routed && sudo systemctl start s2s`.

## Making changes on the server

Just SSH in. The pip install is a wheel; to hack the source:

```bash
git clone https://github.com/huggingface/speech-to-speech ~/src
source ~/s2s/bin/activate
pip install -e "$HOME/src/[faster-whisper,kokoro]"   # editable install
sudo systemctl restart s2s        # or s2s-routed, whichever is active
```

## If something breaks

- `journalctl -u s2s -n 50` — server logs (pure CPU)
- `journalctl -u s2s-routed -n 50` — server logs (hybrid)
- `journalctl -u s2s-tools -n 50` — tool API logs
- `curl localhost:8766/run-tool -X POST -d '{"name":"get_claim_result","arguments":{"search_id":"x"}}'`
  → `{"status":"error",...}` means the tool API is reachable
