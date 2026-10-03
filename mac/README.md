# Mac client

The Mac is the thin client: microphone + speaker + tool menu. All heavy work
runs on the server (Colab GPU box or Oracle CPU box).

## Setup (one time)

```bash
python3.11 -m venv ~/Desktop/s2s-client
source ~/Desktop/s2s-client/bin/activate
pip install speech-to-speech
# macOS SSL fix for python.org Python (if needed):
/Applications/Python\ 3.11/Install\ Certificates.command
```

Save `slow_tools.py` next to the venv (e.g. `~/Desktop/`).

## Talk — via Colab (GPU)

```bash
source ~/Desktop/s2s-client/bin/activate
export OPENAI_API_KEY=not-needed
export TOOL_API_URL=<from Colab tunnel cell>
PYTHONPATH=$HOME/Desktop speech-to-speech talk \
  --url wss://<your-ngrok-domain>/v1/realtime \
  --playback-buffer-ms 800 \
  --block-mic-during-playback \
  --tool-module slow_tools \
  --instructions "You are a helpful voice assistant for an insurance company. When the user asks about a claim: speak 'Let me start that lookup for you.', call lookup_claim_status, then tell the user the search is running in the background (about 10 seconds) and they can ask about anything else meanwhile. When they follow up, call get_claim_result with the search_id. Never invent claim details."
```

## Talk — via Oracle (CPU, 24/7, no tunnel)

```bash
source ~/Desktop/s2s-client/bin/activate
export OPENAI_API_KEY=not-needed
export TOOL_API_URL=http://<ORACLE-PUBLIC-IP>:8766
PYTHONPATH=$HOME/Desktop speech-to-speech talk \
  --url ws://<ORACLE-PUBLIC-IP>:8765/v1/realtime \
  --playback-buffer-ms 800 \
  --block-mic-during-playback \
  --tool-module slow_tools \
  --instructions "You are a helpful voice assistant for an insurance company. ..."
```

Notes:
- `OPENAI_API_KEY=not-needed` — the SDK requires a non-empty key; our server ignores it.
- `PYTHONPATH` must point at the directory containing `slow_tools.py`
  (console scripts don't add the cwd to the import path).
- `--block-mic-during-playback` prevents the mic hearing the reply (kills barge-in;
  headphones are the better fix if you want barge-in).
