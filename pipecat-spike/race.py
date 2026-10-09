"""Phase 3, step 2: Akhil's GPU/CPU race as Pipecat processors.

Ports the signature race design from oracle/router_plugin.py into Pipecat
frame processors:

- RacingSTT(SegmentedSTTService): RunPod Parakeet (GPU) vs faster-whisper
  small.en int8 (CPU). First finisher wins.
- RacingTTS(TTSService): RunPod Qwen3-TTS clone (GPU) vs Supertonic 3 (CPU).
  First finisher wins.

Race rules (mirroring production):
- GPU legs get GPU_ROUTE_TIMEOUT_S (default 8s). A cold start takes ~60s,
  so a timeout means "GPU is cold", NOT "GPU is broken": the CPU wins, and
  the timeout is never counted toward the circuit breaker.
- Real GPU errors DO count: GPU_ROUTE_MAX_FAILURES (default 3) consecutive
  real errors open the breaker for GPU_ROUTE_COOLDOWN_S (default 300s),
  during which turns go pure CPU with zero added latency.
- Warm shortcut: if the GPU won a race quickly (< GPU_WARM_TIMEOUT_S,
  default 5s), subsequent turns skip the CPU leg entirely (no background
  CPU burn). Idle longer than GPU_WARM_IDLE_RESET_S (default 240s) marks
  the endpoint cold again.
- The losing leg keeps running in the background (warms the worker), same
  as production's thread-pool race.

Env knobs (same names as production):
    GPU_ROUTE_TIMEOUT_S        per-attempt GPU cap, seconds (default 8)
    GPU_ROUTE_MAX_FAILURES     real GPU errors before cooldown (default 3)
    GPU_ROUTE_COOLDOWN_S       cooldown after tripping, seconds (default 300)
    GPU_WARM_TIMEOUT_S         warm-GPU cap, seconds (default 5)
    GPU_WARM_IDLE_RESET_S      idle seconds before warm resets (default 240)
    RUNPOD_API_KEY             RunPod API key
    RUNPOD_STT_ENDPOINT_ID     Parakeet endpoint (default acuml4hbia4g1f)
    RUNPOD_TTS_CLONE_ENDPOINT_ID  Qwen clone endpoint (default naq5rfqu0i3g7m)
    SUPERTONIC_VOICE           CPU TTS voice (default F1)
    SUPERTONIC_STEPS           CPU TTS diffusion steps (default 6)
    SUPERTONIC_STYLE_PATH      optional Voice Builder JSON for cloned CPU voice

NOT in scope (later steps): LangGraph loop (step 4), transport.
Voice selection (step 3) is implemented: RacingTTS accepts an optional
voice_resolver (see voice.py) for dynamic per-turn voice switching on the
GPU leg.
"""

import asyncio
import io
import logging
import os
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService

from spike import RunPodParakeetSTT, RunPodQwenTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipecat-race")

# --- Tunables (same names/defaults as oracle/router_plugin.py) ---
GPU_TIMEOUT_S = float(os.environ.get("GPU_ROUTE_TIMEOUT_S", "8"))
MAX_FAILURES = int(os.environ.get("GPU_ROUTE_MAX_FAILURES", "3"))
COOLDOWN_S = float(os.environ.get("GPU_ROUTE_COOLDOWN_S", "300"))
WARM_TIMEOUT_S = float(os.environ.get("GPU_WARM_TIMEOUT_S", "5"))
WARM_IDLE_RESET_S = float(os.environ.get("GPU_WARM_IDLE_RESET_S", "240"))

STT_ENDPOINT_ID = os.environ.get("RUNPOD_STT_ENDPOINT_ID", "acuml4hbia4g1f")
TTS_ENDPOINT_ID = os.environ.get("RUNPOD_TTS_CLONE_ENDPOINT_ID", "naq5rfqu0i3g7m")


class _ColdGPU(Exception):
    """Internal signal: the GPU leg timed out (cold worker), not a real error."""


class _CircuitBreaker:
    """Skip the GPU leg for COOLDOWN_S after MAX_FAILURES consecutive real errors.

    Timeouts (cold starts) never count -- only genuine errors trip the breaker.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._failures = 0
        self._cooldown_until = 0.0

    def allow(self) -> bool:
        return time.monotonic() >= self._cooldown_until

    def success(self) -> None:
        self._failures = 0

    def failure(self) -> None:
        self._failures += 1
        if self._failures >= MAX_FAILURES:
            self._cooldown_until = time.monotonic() + COOLDOWN_S
            self._failures = 0
            logger.warning(
                "%s: circuit open for %.0fs after %d consecutive GPU errors",
                self.name, COOLDOWN_S, MAX_FAILURES,
            )


class _RaceState:
    """Per-stage race state: breaker, warm flag, last-race timestamp."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.breaker = _CircuitBreaker(f"race-{stage}")
        self.gpu_warm = False
        self.last_race_ts = 0.0

    def is_warm(self) -> bool:
        if not self.gpu_warm:
            return False
        idle = time.monotonic() - self.last_race_ts
        if idle > WARM_IDLE_RESET_S:
            self.gpu_warm = False
            logger.info(
                "%s: GPU marked cold after %.0fs idle (endpoint likely scaled to zero)",
                self.stage, idle,
            )
            return False
        return True


def _silence(t: "asyncio.Task") -> None:
    """Swallow a background task's outcome (the race loser keeps warming)."""
    if not t.cancelled():
        t.exception()  # retrieve to avoid "exception was never retrieved"


# Strong refs for backgrounded loser tasks: keeps them alive until done so
# their _silence callback always runs (no GC-before-callback warnings).
_background_tasks: set["asyncio.Task"] = set()


def _launch_background(t: "asyncio.Task") -> None:
    _background_tasks.add(t)
    t.add_done_callback(_background_tasks.discard)
    t.add_done_callback(_silence)


async def _race_turn(
    state: _RaceState,
    gpu_fn: Callable[[], Awaitable],
    cpu_fn: Callable[[], Awaitable],
) -> tuple[str, object]:
    """Run one turn of STT/TTS; return (winner, result).

    Mirrors oracle/router_plugin._race: warm shortcut when the GPU is warm,
    otherwise fire both legs and take the first finisher. If the first
    finisher raised, the other leg's result is used instead (raises only if
    both failed).

    NOTE: unlike production's _race (which stamps last_race_ts before the
    warm check, making the idle-reset branch unreachable), we check warm
    FIRST so idle > WARM_IDLE_RESET_S genuinely marks the endpoint cold.
    """
    # Check warm BEFORE stamping: idle is measured since the previous race.
    warm = state.is_warm()
    state.last_race_ts = time.monotonic()

    # Breaker open (GPU known-dead): pure CPU, zero added latency.
    if not state.breaker.allow():
        logger.info("%s: breaker open, CPU-only this turn", state.stage)
        return "cpu", await cpu_fn()

    async def guard_gpu():
        """GPU leg with breaker bookkeeping; timeouts are cold, not failures."""
        try:
            result = await asyncio.wait_for(gpu_fn(), timeout=GPU_TIMEOUT_S)
        except (asyncio.TimeoutError, TimeoutError):
            logger.info(
                "%s: GPU leg timed out after %.0fs (cold worker?)",
                state.stage, GPU_TIMEOUT_S,
            )
            raise _ColdGPU()
        except Exception:
            state.breaker.failure()
            logger.warning("%s: GPU leg errored", state.stage, exc_info=True)
            raise
        state.breaker.success()
        return result

    # Warm shortcut: GPU-only, no background CPU burn.
    if warm:
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(gpu_fn(), timeout=WARM_TIMEOUT_S)
            state.breaker.success()
        except (asyncio.TimeoutError, TimeoutError):
            state.gpu_warm = False
            logger.info(
                "%s: warm GPU exceeded %.0fs, CPU fallback this turn",
                state.stage, WARM_TIMEOUT_S,
            )
            return "cpu", await cpu_fn()
        except Exception:
            state.breaker.failure()
            state.gpu_warm = False
            logger.info("%s: warm GPU failed, CPU fallback this turn", state.stage)
            return "cpu", await cpu_fn()
        logger.info(
            "%s won by gpu (warm shortcut) in %.2fs",
            state.stage, time.perf_counter() - started,
        )
        return "gpu", result

    # Cold path: fire both, first finisher wins.
    started = time.perf_counter()
    gpu_task = asyncio.create_task(guard_gpu())
    cpu_task = asyncio.create_task(cpu_fn())
    done, _pending = await asyncio.wait(
        {gpu_task, cpu_task}, return_when=asyncio.FIRST_COMPLETED
    )
    first = next(iter(done))
    losers = [t for t in (gpu_task, cpu_task) if t is not first]
    try:
        result = first.result()
        winner = "gpu" if first is gpu_task else "cpu"
    except Exception:
        other = cpu_task if first is gpu_task else gpu_task
        winner = "cpu" if first is gpu_task else "gpu"
        result = await other  # raises only if both legs failed
    # Silence losers: already finished -> retrieve now; still running ->
    # keep warming in the background, silence on completion.
    for t in losers:
        if t.done():
            _silence(t)
        else:
            _launch_background(t)
    # Warm = GPU won *quickly*; a slow GPU win means the worker is still cold.
    state.gpu_warm = (
        winner == "gpu" and time.perf_counter() - started < WARM_TIMEOUT_S
    )
    logger.info(
        "%s won by %s in %.2fs", state.stage, winner, time.perf_counter() - started
    )
    return winner, result


def _wav_to_f32_mono(wav_bytes: bytes, target_sr: int = 16000):
    """Decode WAV bytes to float32 mono numpy at target_sr (linear resample)."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != target_sr:
        ratio = target_sr / sr
        idx = (np.arange(int(len(data) * ratio)) / ratio).astype(int)
        idx = np.clip(idx, 0, len(data) - 1)
        data = data[idx]
    return data


class RacingSTT(SegmentedSTTService):
    """STT that races RunPod Parakeet (GPU) vs faster-whisper (CPU).

    Reuses RunPodParakeetSTT from spike.py for the GPU leg; the CPU leg is
    faster-whisper small.en int8, lazily loaded (same as production).
    """

    def __init__(self, *, api_key: str = "", endpoint_id: str = "", **kwargs):
        super().__init__(**kwargs)
        self._gpu = RunPodParakeetSTT(
            api_key=api_key or os.environ.get("RUNPOD_API_KEY", ""),
            endpoint_id=endpoint_id or STT_ENDPOINT_ID,
        )
        self._state = _RaceState("stt")
        self._fw_model = None
        self._fw_lock = threading.Lock()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        try:
            winner, text = await _race_turn(
                self._state, lambda: self._gpu_transcribe(audio), lambda: self._cpu_transcribe(audio)
            )
        except Exception as e:
            logger.exception("RacingSTT: both legs failed")
            yield ErrorFrame(error=f"STT failed: {e}")
            return
        logger.info("STT won by %s: %r", winner, text[:120])
        yield TranscriptionFrame(text=text, user_id="", timestamp="")

    async def _gpu_transcribe(self, audio: bytes) -> str:
        """Transcribe via the RunPod GPU leg; returns plain text."""
        text = None
        async for frame in self._gpu.run_stt(audio):
            if isinstance(frame, TranscriptionFrame):
                text = frame.text
            elif isinstance(frame, ErrorFrame):
                raise RuntimeError(frame.error)
        if text is None:
            raise RuntimeError("GPU STT returned no transcript")
        return text

    async def _cpu_transcribe(self, audio: bytes) -> str:
        return await asyncio.to_thread(self._fw_transcribe, audio)

    def _fw_transcribe(self, audio: bytes) -> str:
        with self._fw_lock:
            if self._fw_model is None:
                from faster_whisper import WhisperModel

                self._fw_model = WhisperModel("small.en", compute_type="int8")
                logger.info("faster-whisper small.en int8 ready (CPU STT leg)")
        pcm = _wav_to_f32_mono(audio, target_sr=16000)
        segments, _ = self._fw_model.transcribe(pcm, language="en")
        return "".join(s.text for s in segments).strip()


class RacingTTS(TTSService):
    """TTS that races RunPod Qwen3-TTS clone (GPU) vs Supertonic 3 (CPU).

    Reuses RunPodQwenTTS from spike.py for the GPU leg; the CPU leg is
    Supertonic 3 ONNX on CPU, lazily loaded (same as production).

    Voice selection (Phase 3, step 3): pass voice_resolver=VoiceResolver()
    and the GPU leg clones whichever voice the UI selected
    (~/voice-profiles/.active_voice), switching mid-session with no restart.
    The CPU leg keeps its fixed voice (SUPERTONIC_VOICE / SUPERTONIC_STYLE_PATH)
    -- Supertonic needs a pre-built style, it can't do per-request ICL.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        endpoint_id: str = "",
        ref_audio_b64: str = "",
        ref_text: str = "",
        sample_rate: int = 16000,
        voice_resolver=None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._gpu = RunPodQwenTTS(
            api_key=api_key or os.environ.get("RUNPOD_API_KEY", ""),
            endpoint_id=endpoint_id or TTS_ENDPOINT_ID,
            ref_audio_b64=ref_audio_b64,
            ref_text=ref_text,
            sample_rate=sample_rate,
            voice_resolver=voice_resolver,
        )
        self._state = _RaceState("tts")
        self._out_sample_rate = sample_rate or 16000
        self._supertonic = None
        self._supertonic_style = None
        self._st_lock = threading.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.start_ttfb_metrics()
        # TTSStartedFrame is pushed by base class _push_tts_frames
        try:
            winner, pcm = await _race_turn(
                self._state, lambda: self._gpu_synthesize(text, context_id), lambda: self._cpu_synthesize(text)
            )
            logger.info("TTS won by %s: %d chars -> %d PCM bytes", winner, len(text), len(pcm))
            # ~20ms chunks (640 bytes @ 16kHz mono 16-bit).
            for i in range(0, len(pcm), 640):
                yield TTSAudioRawFrame(
                    audio=pcm[i : i + 640],
                    sample_rate=self._out_sample_rate,
                    num_channels=1,
                )
        except Exception as e:
            logger.exception("RacingTTS: both legs failed")
            yield ErrorFrame(error=f"TTS failed: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    async def _gpu_synthesize(self, text: str, context_id: str) -> bytes:
        """Synthesize via the RunPod GPU leg; returns raw PCM16 bytes."""
        chunks: list[bytes] = []
        async for frame in self._gpu.run_tts(text, context_id):
            if isinstance(frame, TTSAudioRawFrame):
                chunks.append(frame.audio)
            elif isinstance(frame, ErrorFrame):
                raise RuntimeError(frame.error)
        pcm = b"".join(chunks)
        if not pcm:
            raise RuntimeError("GPU TTS returned no audio")
        return pcm

    async def _cpu_synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._st_synthesize, text)

    def _st_synthesize(self, text: str) -> bytes:
        """Synthesize via Supertonic 3 on CPU; returns raw PCM16 bytes @16kHz."""
        import numpy as np
        from math import gcd
        from scipy.signal import resample_poly

        with self._st_lock:
            if self._supertonic is None:
                from supertonic import TTS

                self._supertonic = TTS(auto_download=True)
                style_path = os.environ.get("SUPERTONIC_STYLE_PATH")
                if style_path:
                    self._supertonic_style = self._supertonic.get_voice_style_from_path(style_path)
                    logger.info("Supertonic 3 ready (custom style: %s)", style_path)
                else:
                    voice = os.environ.get("SUPERTONIC_VOICE", "F1")
                    self._supertonic_style = self._supertonic.get_voice_style(voice_name=voice)
                    logger.info("Supertonic 3 ready (built-in voice: %s)", voice)
        steps = int(os.environ.get("SUPERTONIC_STEPS", "6"))
        wav, _duration = self._supertonic.synthesize(
            text=text, voice_style=self._supertonic_style, total_steps=steps, lang="en"
        )
        # float32 (1, N) @ 44100 Hz -> int16 PCM bytes @ 16kHz.
        audio = np.asarray(wav).squeeze().astype(np.float32)
        g = gcd(44100, 16000)
        audio = resample_poly(audio, 16000 // g, 44100 // g).astype(np.float32)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm16.tobytes()
