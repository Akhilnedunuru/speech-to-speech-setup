# Voice Cloning UI (Phase 2)

A simple web app to clone a voice and hear it speak. Built for the
speech-to-speech project — it calls the RunPod Qwen3-TTS clone endpoint
directly (no GPU/CPU race, no fallback), so you always hear the cloned voice.

## How it works

- **`/` (Clone)** — Read a paragraph into your mic. The browser records it,
  converts to 16kHz mono WAV, and the server auto-transcribes it via the
  Parakeet STT endpoint. The profile (audio + transcript) is saved to
  `~/voice-profiles/<profile-id>/`.
- **`/try` (Try it)** — Pick a saved voice, type text, get audio back in that
  voice. If the GPU endpoint is cold, you'll see a "warming up" note while
  the request waits (~60s first time, ~2s after).

## Run on the VM

```bash
cd ~/speech-to-speech-setup
git pull
cd voice-ui
~/s2s/bin/pip install -r requirements.txt

export RUNPOD_API_KEY="your-runpod-key"
# optional overrides (defaults shown):
export RUNPOD_STT_ENDPOINT_ID="acuml4hbia4g1f"
export RUNPOD_TTS_CLONE_ENDPOINT_ID="naq5rfqu0i3g7m"
export VOICE_UI_PORT="8080"

~/s2s/bin/python app.py
```

Then open `http://<VM-IP>:8080` in a browser.
(Chrome/Edge recommended — mic recording needs a secure context; use
`localhost` or HTTPS for `getUserMedia`. Over plain HTTP on a remote IP,
browsers block mic access. For public sharing, put Caddy in front with a
DuckDNS domain.)

## Notes

- Profiles are plain directories under `~/voice-profiles/` — the same layout
  the voice pipeline uses, so a UI-created profile works with the pipeline too.
- Rate limit: 10 requests/min per IP (in-memory).
- Synthesis text cap: 500 chars per request.
- No auth in v1 — don't expose to the open internet without a reverse proxy.
