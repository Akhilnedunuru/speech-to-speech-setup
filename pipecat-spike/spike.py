"""Pipecat spike (Phase 3, step 1): prove Pipecat frames flow through custom
processors that call Akhil's existing RunPod endpoints.

Pipeline:
    VAD (Silero) → RunPodParakeetSTT → OpenAILLMService (Groq) → RunPodQwenTTS

Custom processors (the point of the spike):
    - RunPodParakeetSTT(SegmentedSTTService): VAD-bounded audio → RunPod
      Parakeet endpoint → TranscriptionFrame
    - RunPodQwenTTS(TTSService): text → RunPod Qwen3-TTS clone endpoint
      → TTSAudioRawFrame

NOT custom (deliberate):
    - LLM uses Pipecat's dedicated GroqLLMService (ships with pipecat-ai[groq]).
      Writing a custom LLMService would mean reimplementing context aggregation;
      the built-in is the idiomatic Pipecat pattern.

What this spike does NOT include (later steps):
    - GPU/CPU race (that's step 2)
    - Voice selection / .active_voice (step 3)
    - LangGraph reasoning loop (step 4)
    - Any transport (WebSocket etc.) — the test harness drives frames directly.

Env vars:
    RUNPOD_API_KEY              RunPod API key
    RUNPOD_STT_ENDPOINT_ID      Parakeet STT endpoint (default: acuml4hbia4g1f)
    RUNPOD_TTS_CLONE_ENDPOINT_ID  Qwen3-TTS clone endpoint (default: naq5rfqu0i3g7m)
    GROQ_API_KEY                Groq API key
    GROQ_MODEL                  Groq model (default: openai/gpt-oss-20b)
    REF_AUDIO_PATH              Path to reference WAV for TTS cloning
    REF_TEXT_PATH               Path to reference transcript text file
"""

import asyncio
import base64
import io
import logging
import os
from collections.abc import AsyncGenerator

import requests

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipecat-spike")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
STT_ENDPOINT_ID = os.environ.get("RUNPOD_STT_ENDPOINT_ID", "acuml4hbia4g1f")
TTS_ENDPOINT_ID = os.environ.get("RUNPOD_TTS_CLONE_ENDPOINT_ID", "naq5rfqu0i3g7m")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# RunPod synchronous call timeout. Covers a full cold start (~60s) + inference.
RUNPOD_TIMEOUT_S = 180


def _runsync(endpoint_id: str, payload: dict) -> dict:
    """POST to a RunPod serverless endpoint, polling if the job is queued.

    Returns the job's output dict. Raises RuntimeError on failure/timeout.
    """
    import time

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    deadline = time.time() + RUNPOD_TIMEOUT_S

    resp = requests.post(
        f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
        headers=headers,
        json={"input": payload},
        timeout=RUNPOD_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()

    status = data.get("status")
    if status == "COMPLETED":
        return _check_output(data)
    if status == "FAILED":
        raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown')}"[:200])
    if status not in ("IN_QUEUE", "IN_PROGRESS"):
        raise RuntimeError(f"RunPod job did not complete: {status}")

    job_id = data.get("id")
    if not job_id:
        raise RuntimeError("RunPod job queued but no job id returned")

    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise RuntimeError(f"RunPod job {job_id} timed out after {RUNPOD_TIMEOUT_S}s")
        time.sleep(min(2, remaining))
        poll = requests.get(status_url, headers=headers, timeout=max(1, min(30, remaining)))
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status")
        if status == "COMPLETED":
            return _check_output(data)
        if status == "FAILED":
            raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown')}"[:200])


def _check_output(data: dict) -> dict:
    output = data.get("output")
    if not isinstance(output, dict):
        raise RuntimeError(f"Unexpected RunPod output: {output!r}"[:200])
    return output


class RunPodParakeetSTT(SegmentedSTTService):
    """STT via Akhil's RunPod Parakeet endpoint.

    SegmentedSTTService handles VAD-bounded audio buffering and calls
    run_stt(audio) with WAV bytes (16-bit mono PCM) when the user stops
    speaking. We just POST the WAV to RunPod and yield a TranscriptionFrame.
    """

    def __init__(self, *, api_key: str, endpoint_id: str, **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._endpoint_id = endpoint_id

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Transcribe WAV bytes via RunPod. Yields a single TranscriptionFrame."""
        global RUNPOD_API_KEY
        old_key, RUNPOD_API_KEY = RUNPOD_API_KEY, self._api_key
        try:
            # Run the blocking HTTP call in a thread so we don't stall the loop.
            output = await asyncio.to_thread(
                _runsync,
                self._endpoint_id,
                {"audio": base64.b64encode(audio).decode("ascii")},
            )
        except Exception as e:
            logger.exception("RunPod STT failed")
            yield ErrorFrame(error=f"STT failed: {e}")
            return
        finally:
            RUNPOD_API_KEY = old_key

        text = (output.get("text") or "").strip()
        logger.info("STT transcript: %r", text[:120])
        yield TranscriptionFrame(text=text, user_id="", timestamp="")


class RunPodQwenTTS(TTSService):
    """TTS via Akhil's RunPod Qwen3-TTS clone endpoint.

    Implements run_tts(text): POST text + reference voice to RunPod,
    decode the returned WAV to raw PCM, yield TTSAudioRawFrames.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint_id: str,
        ref_audio_b64: str,
        ref_text: str,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._endpoint_id = endpoint_id
        self._ref_audio_b64 = ref_audio_b64
        self._ref_text = ref_text
        # TTSService stores sample_rate as self._sample_rate (set in setup());
        # keep our own copy for use in run_tts before setup() runs.
        self._out_sample_rate = sample_rate or 16000

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        global RUNPOD_API_KEY
        old_key, RUNPOD_API_KEY = RUNPOD_API_KEY, self._api_key
        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            output = await asyncio.to_thread(
                _runsync,
                self._endpoint_id,
                {
                    "text": text,
                    "ref_audio": self._ref_audio_b64,
                    "ref_text": self._ref_text,
                    "language": "english",
                },
            )
            audio_b64 = output.get("audio")
            if not audio_b64:
                raise RuntimeError("Clone endpoint returned no audio")

            pcm = _wav_to_pcm16(base64.b64decode(audio_b64), target_sr=self._out_sample_rate)
            logger.info("TTS: %d chars -> %d PCM bytes", len(text), len(pcm))

            # Yield in ~20ms chunks (320 samples @ 16kHz mono 16-bit = 640 bytes).
            chunk_bytes = 640
            for i in range(0, len(pcm), chunk_bytes):
                yield TTSAudioRawFrame(
                    audio=pcm[i : i + chunk_bytes],
                    sample_rate=self._out_sample_rate,
                    num_channels=1,
                )
        except Exception as e:
            logger.exception("RunPod TTS failed")
            yield ErrorFrame(error=f"TTS failed: {e}")
        finally:
            await self.stop_ttfb_metrics()
            RUNPOD_API_KEY = old_key
            yield TTSStoppedFrame()


def _wav_to_pcm16(wav_bytes: bytes, target_sr: int = 16000) -> bytes:
    """Decode a WAV file to raw 16-bit mono PCM at target_sr.

    Uses soundfile for robust decoding; resamples with numpy if needed.
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16", always_2d=True)
    mono = data[:, 0]  # take first channel
    if sr != target_sr:
        # Linear resample (good enough for a spike; use librosa/scipy in prod).
        ratio = target_sr / sr
        idx = (np.arange(int(len(mono) * ratio)) / ratio).astype(int)
        idx = np.clip(idx, 0, len(mono) - 1)
        mono = mono[idx]
    return mono.astype("<i2").tobytes()


def build_services(ref_audio_b64: str, ref_text: str):
    """Construct the STT / LLM / TTS services. Returns (stt, llm, tts)."""
    stt = RunPodParakeetSTT(api_key=RUNPOD_API_KEY, endpoint_id=STT_ENDPOINT_ID)

    # Pipecat ships a dedicated Groq service (OpenAI-compatible). No custom
    # LLM code needed — this is the idiomatic pattern.
    # NOTE: pass model via Settings (the plain `model=` kwarg is deprecated
    # since 0.0.105 and warns). reasoning_effort="low" matches production.
    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        settings=GroqLLMService.Settings(model=GROQ_MODEL, reasoning_effort="low"),
    )

    tts = RunPodQwenTTS(
        api_key=RUNPOD_API_KEY,
        endpoint_id=TTS_ENDPOINT_ID,
        ref_audio_b64=ref_audio_b64,
        ref_text=ref_text,
    )
    return stt, llm, tts


def load_reference_voice() -> tuple[str, str]:
    """Load reference WAV + transcript, return (base64_wav, text)."""
    ref_audio_path = os.environ.get("REF_AUDIO_PATH", "")
    ref_text_path = os.environ.get("REF_TEXT_PATH", "")
    if not ref_audio_path or not ref_text_path:
        raise RuntimeError("Set REF_AUDIO_PATH and REF_TEXT_PATH env vars")
    with open(ref_audio_path, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode("ascii")
    with open(ref_text_path) as f:
        ref_text = f.read().strip()
    return ref_b64, ref_text


async def main():
    """Wire the pipeline (no transport in the spike) and report what was built."""
    ref_b64, ref_text = load_reference_voice()
    stt, llm, tts = build_services(ref_b64, ref_text)
    logger.info(
        "Spike services constructed: %s, %s, %s",
        type(stt).__name__,
        type(llm).__name__,
        type(tts).__name__,
    )
    logger.info(
        "Pipeline order: SileroVAD → RunPodParakeetSTT → OpenAILLMService(Groq) → RunPodQwenTTS"
    )
    logger.info("See test_processors.py for mocked processor tests.")


if __name__ == "__main__":
    asyncio.run(main())
