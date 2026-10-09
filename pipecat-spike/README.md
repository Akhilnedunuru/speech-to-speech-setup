# Pipecat Spike (Phase 3, step 1)

Proof-of-concept: Pipecat frames flowing through custom processors that call
Akhil's existing RunPod endpoints. This validates the frame model before the
full rebuild.

## What it proves

- `RunPodParakeetSTT(SegmentedSTTService)` — VAD-bounded audio → RunPod
  Parakeet HTTP endpoint → `TranscriptionFrame`. The base class handles audio
  buffering between speech-start/stop; we only implement `run_stt(audio)`.
- `RunPodQwenTTS(TTSService)` — text → RunPod Qwen3-TTS clone endpoint →
  `TTSAudioRawFrame` chunks. Implements `run_tts(text)`.
- LLM uses Pipecat's **dedicated** `GroqLLMService` (from `pipecat-ai[groq]`).
  Writing a custom LLMService would mean reimplementing context aggregation;
  the built-in is the idiomatic pattern.

## What's NOT here (later steps)

- **Step 2**: GPU/CPU race → custom Pipecat processors with fallback
- **Step 3**: Voice selection (`.active_voice`) → runtime profile switching
- **Step 4**: LangGraph reasoning loop inside the LLM stage
- **Transport**: no WebSocket/WebRTC yet — the test harness drives frames
  directly. `FastAPIWebsocketTransport` is the intended transport later
  (note: `WebsocketServerTransport` is deprecated as of Pipecat 1.4).

## Run

```bash
cd pipecat-spike
pip install -r requirements.txt

export RUNPOD_API_KEY="..."
export GROQ_API_KEY="..."
# optional overrides:
export RUNPOD_STT_ENDPOINT_ID="acuml4hbia4g1f"
export RUNPOD_TTS_CLONE_ENDPOINT_ID="naq5rfqu0i3g7m"
export GROQ_MODEL="openai/gpt-oss-20b"
export REF_AUDIO_PATH="/path/to/reference.wav"   # for TTS cloning
export REF_TEXT_PATH="/path/to/ref_text.txt"

# Mocked processor tests (no network):
python test_processors.py

# Construct services (needs env vars, no network until frames flow):
python spike.py
```

## API notes / surprises

1. **`SegmentedSTTService`** (in `pipecat.services.stt_service`) is the right
   base for non-streaming HTTP STT like Parakeet. It buffers audio between
   `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame` and calls
   `run_stt(audio: bytes)` with WAV bytes. `TranscriptionFrame` is marked
   finalized automatically.
2. **`TTSService.run_tts`** yields `TTSStartedFrame`, then `TTSAudioRawFrame`
   chunks (raw PCM, not WAV), then `TTSStoppedFrame`. The base class handles
   sentence aggregation; we decode the endpoint's WAV response to PCM.
3. **Deprecations to avoid**: `PipelineTask` → `PipelineWorker`,
   `PipelineRunner` → `WorkerRunner`, `WebsocketServerTransport` (1.4.0).
   The spike avoids all three.
4. **Qwen language quirk**: the clone endpoint expects `"english"`, not
   `"en"` (learned the hard way in Phase 2).
5. **RunPod `/runsync`** can return `IN_QUEUE` when the endpoint is cold;
   `_runsync()` polls `/status/{job_id}` until `COMPLETED` (same pattern as
   the Phase 2 web UI).
