# Free 24/7 voice server on Oracle Cloud (Always Free)

No free GPU stays up 24/7 — Colab/Kaggle are ephemeral by design. So this setup
moves to the only big-cloud VM that's free *forever* and big enough: Oracle's
Always Free Ampere box (4 ARM cores, 24 GB RAM). It's CPU-only, so the two
GPU-hungry stages get swapped for CPU-friendly ones:

| Stage | Colab (T4) | Oracle (Ampere CPU) |
|---|---|---|
| VAD / Smart Turn | CPU | CPU (unchanged) |
| STT | Parakeet (GPU) | **faster-whisper `small.en`, int8 (CPU)** |
| LLM | Groq via HF router | Groq via HF router (unchanged, still free) |
| TTS | Qwen3-TTS (GPU) | **Kokoro 82M (CPU)** |
| Tools | `tool_server.py` on Colab | `tool_server.py` on Oracle (unchanged code) |
| Public URL | ngrok tunnel | **none needed — the VM has a public IP** |

Honest latency expectation: roughly 4–8s per turn (vs ~3–5s on the T4).
Fine for learning and dev; not production-grade.

## Step 1 — Oracle account

1. Sign up at cloud.oracle.com (Always Free tier). A credit card is required
   **for verification only** — the Always Free resources are not charged.
2. Pick a home region near Texas: **us-phoenix-1** first; if Ampere is
   "out of capacity" (common), try **us-ashburn-1**.

## Step 2 — Create the VM

Compute → Instances → Create:
- Image: **Ubuntu 24.04**, shape **VM.Standard.A1.Flex** (Ampere ARM)
- OCPUs: **4**, Memory: **24 GB**, Boot volume: 100 GB (all within Always Free)
- Add your SSH public key (`~/.ssh/id_ed25519.pub` on your Mac)
- Note the **public IP** after it boots

## Step 3 — Open ports

The VM has a public IP, so no ngrok/tunnel needed. Open the two ports:

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

## Step 5 — Token + services

```bash
echo 'HF_TOKEN=<your-hf-token>' > ~/.s2s_env && chmod 600 ~/.s2s_env
sudo cp ~/s2s.service ~/s2s-tools.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now s2s s2s-tools
systemctl status s2s            # should be active
journalctl -u s2s -f            # watch startup logs (~1-2 min first boot)
```

systemd restarts both services if they crash, and they come back up on VM reboot.
That's your 24/7.

## Step 6 — Mac client (no tunnel anymore!)

```bash
cd ~/Desktop
source s2s-client/bin/activate
export OPENAI_API_KEY=not-needed
export TOOL_API_URL=http://<PUBLIC-IP>:8766
PYTHONPATH=$HOME/Desktop speech-to-speech talk \
  --url ws://<PUBLIC-IP>:8765/v1/realtime \
  --playback-buffer-ms 800 \
  --block-mic-during-playback \
  --tool-module slow_tools \
  --instructions "You are a helpful voice assistant for an insurance company. ..."
```

Note `ws://` (not `wss://`) — no TLS on the raw IP, which is fine for personal dev.
(If you later want `wss://`, point a free DuckDNS subdomain at the IP and put Caddy
in front for automatic HTTPS.)

## Making changes on the server

Just SSH in. The pip install is a wheel; to hack the source:

```bash
git clone https://github.com/huggingface/speech-to-speech ~/src
source ~/s2s/bin/activate
pip install -e " ~/src[ faster-whisper,kokoro ] "  # editable install
sudo systemctl restart s2s
```

## If something breaks

- `journalctl -u s2s -n 50` — server logs
- `journalctl -u s2s-tools -n 50` — tool API logs
- `curl localhost:8766/run-tool -X POST -d '{"name":"get_claim_result","arguments":{"search_id":"x"}}'`
  → `{"status":"error",...}` means the tool API is reachable
