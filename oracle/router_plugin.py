"""
GPU-routed STT/TTS backends for the Oracle 24/7 box.

Registers two new backends with the speech-to-speech backend registry:

    --stt runpod-routed    Parakeet TDT on RunPod serverless, raced with faster-whisper CPU
    --tts runpod-routed    Qwen3-TTS on RunPod serverless, raced with Kokoro CPU

Per-turn behavior (race, first finisher wins):
  1. If the circuit breaker allows, fire BOTH the RunPod GPU request and the
     local CPU inference at once; whichever finishes first answers the turn.
  2. Cold GPU (slow) -> CPU wins with zero added latency. The abandoned GPU
     request keeps warming the worker in the background, so the NEXT turns
     hit a warm GPU.
  3. Warm GPU (fast) -> GPU wins: better quality, and the background CPU
     inference is simply discarded.
  4. Only real GPU errors count toward the circuit breaker. A timeout just
     means "cold", not "broken", so cold starts can never trip the breaker.
  5. If the breaker is open (GPU known-dead), the GPU leg is skipped for
     GPU_ROUTE_COOLDOWN_S and turns go pure CPU with zero added latency.
  6. Warm shortcut: once the GPU wins a race quickly, the next turns skip
     the CPU leg entirely (GPU-only) so losing CPU inference stops burning
     the box's cores in the background. A stumble falls back to CPU for
     that turn and re-arms the full race; idleness past the endpoint's
     scale-to-zero window resets to cold.

The router subclasses the stock CPU handlers, so setup(), config flags and
the queue/thread contract are inherited unchanged. Only process() is overridden.

Env:
  RUNPOD_API_KEY            RunPod API key (required)
  RUNPOD_STT_ENDPOINT_ID    serverless endpoint id for STT (required for --stt runpod-routed)
  RUNPOD_TTS_ENDPOINT_ID    serverless endpoint id for TTS (required for --tts runpod-routed)
  GPU_ROUTE_TIMEOUT_S       per-attempt cap on the GPU request, seconds (default 8).
                            No longer user-facing: the race, not the timeout,
                            decides each turn. Only bounds a wedged worker.
  GPU_ROUTE_MAX_FAILURES    real GPU errors before cooldown (default 3)
  GPU_ROUTE_COOLDOWN_S      cooldown after tripping, seconds (default 300)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, wait
from typing import Any, Callable, Iterator

import numpy as np
import requests
import soundfile as sf

from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler
from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
from speech_to_speech.pipeline.handler_types import STTIn, STTOut, TTSIn, TTSOut
from speech_to_speech.pipeline.messages import EndOfResponse, Transcription

logger = logging.getLogger(__name__)

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
STT_ENDPOINT_ID = os.environ.get("RUNPOD_STT_ENDPOINT_ID", "")
TTS_ENDPOINT_ID = os.environ.get("RUNPOD_TTS_ENDPOINT_ID", "")
TIMEOUT_S = float(os.environ.get("GPU_ROUTE_TIMEOUT_S", "8"))
MAX_FAILURES = int(os.environ.get("GPU_ROUTE_MAX_FAILURES", "3"))
COOLDOWN_S = float(os.environ.get("GPU_ROUTE_COOLDOWN_S", "300"))
WARM_TIMEOUT_S = float(os.environ.get("GPU_WARM_TIMEOUT_S", "5"))
WARM_IDLE_RESET_S = float(os.environ.get("GPU_WARM_IDLE_RESET_S", "240"))

# Whisper-style language code -> Qwen3-TTS language name (mirrors upstream aliases)
CODE_TO_QWEN3_LANG = {
    "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean",
    "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese",
    "es": "spanish", "it": "italian", "hi": "hindi",
}

TTS_CHUNK_SAMPLES = 1600  # 100 ms @ 16 kHz

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split reply text into small clauses (~80 chars) for streaming TTS.

    Finer chunks = faster first sound and better generation/playback overlap.
    Per-request overhead is small vs inference (~27 chars/s), so this is
    nearly free.
    """
    sentences = [p for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p and p.strip()]
    chunks: list[str] = []
    for sent in sentences:
        # Further split long sentences on clause boundaries (comma/semicolon/colon)
        while len(sent) > 80:
            m = None
            for mm in re.finditer(r'[,;:]\s+', sent):
                if mm.end() <= 80:
                    m = mm
                else:
                    break
            if m:
                chunks.append(sent[:m.end()].strip())
                sent = sent[m.end():].strip()
            else:
                # No clause boundary: hard split at 80 chars on a word boundary
                sp = sent.rfind(' ', 0, 80)
                cut = sp if sp > 40 else 80
                chunks.append(sent[:cut].strip())
                sent = sent[cut:].strip()
        if sent:
            chunks.append(sent)
    return chunks

# Shared pool for the race: at most one race per stage is ever in flight,
# each race needs 2 threads (gpu + cpu).
_race_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gpu-race")


class CircuitBreaker:
    """Skip the GPU leg for COOLDOWN_S after MAX_FAILURES consecutive real errors."""

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


_stt_breaker = CircuitBreaker("runpod-stt")
_tts_breaker = CircuitBreaker("runpod-tts")

# Warm shortcut: per-stage flag set when the GPU wins a race quickly, cleared
# when the GPU loses, errors, or the endpoint has likely scaled to zero (idle
# longer than the RunPod idle timeout). While warm, the CPU leg is skipped
# entirely so losing CPU inference stops burning the box's 2 cores in the
# background and starving the next turn's pipeline.
_gpu_warm = {"stt": False, "tts": False}
_last_race_ts = {"stt": 0.0, "tts": 0.0}


def _is_warm(stage: str) -> bool:
    if not _gpu_warm[stage]:
        return False
    idle = time.monotonic() - _last_race_ts[stage]
    if idle > WARM_IDLE_RESET_S:
        _gpu_warm[stage] = False
        logger.info("%s: GPU marked cold after %.0fs idle (endpoint likely scaled to zero)",
                    stage, idle)
        return False
    return True


def _runsync(endpoint_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to a RunPod serverless endpoint, synchronously. Raises on any failure."""
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY is not set")
    if not endpoint_id:
        raise RuntimeError("RunPod endpoint id is not set")
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={"input": payload},
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "COMPLETED":
        raise RuntimeError(f"RunPod status={data.get('status')}: {str(data.get('error'))[:200]}")
    return data["output"]


def _wav_b64(audio: np.ndarray, sample_rate: int = 16000) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _wav_b64_to_int16(wav_b64: str) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(base64.b64decode(wav_b64)), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"TTS endpoint returned {sr} Hz, expected 16000")
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def _guard_gpu(name: str, breaker: CircuitBreaker, gpu_fn: Callable[[], list]) -> list:
    """Run the GPU leg with breaker bookkeeping.

    A timeout means the worker is cold, not broken: it is logged, re-raised
    (so the race treats it as "lost"), and never counted as a failure.
    """
    try:
        result = gpu_fn()
    except requests.Timeout:
        logger.info("%s: GPU leg timed out after %.0fs (cold worker?)", name, TIMEOUT_S)
        raise
    except Exception:
        breaker.failure()
        logger.warning("%s: GPU leg errored", name, exc_info=True)
        raise
    breaker.success()
    return result


# Cross-endpoint warmup: the first STT race fires a background TTS warmup
# and vice versa, so the other endpoint's cold start overlaps this turn
# instead of stacking behind it. Only warms the *other* endpoint to avoid
# queue-blocking real requests (max_workers=1).
_warmup_lock = threading.Lock()
_warmup_fired_for: set = set()


def _silent_wav_b64() -> str:
    """A 0.5s silent 16kHz mono WAV, base64-encoded, for STT warmup."""
    import wave
    import struct
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<8000h", *([0] * 8000)))
    return base64.b64encode(buf.getvalue()).decode()


def _fire_cross_warmup(stage: str) -> None:
    with _warmup_lock:
        if stage in _warmup_fired_for:
            return
        _warmup_fired_for.add(stage)
    other = "tts" if stage == "stt" else "stt"

    def _do() -> None:
        try:
            if other == "tts":
                _runsync(TTS_ENDPOINT, {"input": {"text": "warmup"}}, timeout=60)
            else:
                _runsync(STT_ENDPOINT, {"input": {"audio": _silent_wav_b64()}}, timeout=60)
            logger.info("warmup: %s endpoint warmed", other)
        except Exception as e:
            logger.info("warmup: %s endpoint warmup failed: %s", other, e)

    threading.Thread(target=_do, daemon=True, name=f"warmup-{other}").start()


def _race(stage: str, breaker: CircuitBreaker,
          gpu_fn: Callable[[], list], cpu_fn: Callable[[], list]) -> tuple[str, list]:
    """Run one turn of STT/TTS; return (winner, result).

    Warm shortcut: if the GPU won the previous race quickly and the endpoint
    has not been idle long enough to scale to zero, the CPU leg is skipped
    entirely -- GPU-only, no background CPU burn. If the warm GPU stumbles,
    this turn falls back to CPU and the next turn races both legs again.

    Cold path: fire GPU and CPU at once, first finisher wins. Cold GPU (slow)
    -> CPU wins with zero added latency while the GPU request keeps warming
    the worker in the background. If the first finisher raised, the other
    leg's result is used instead.
    """
    _last_race_ts[stage] = time.monotonic()
    _fire_cross_warmup(stage)

    if _is_warm(stage) and breaker.allow():
        started = time.perf_counter()
        fut = _race_pool.submit(_guard_gpu, stage, breaker, gpu_fn)
        try:
            result = fut.result(timeout=WARM_TIMEOUT_S)
        except FuturesTimeoutError:
            _gpu_warm[stage] = False
            logger.info("%s: warm GPU exceeded %.0fs, CPU fallback this turn",
                        stage, WARM_TIMEOUT_S)
            return "cpu", cpu_fn()
        except Exception:
            _gpu_warm[stage] = False
            logger.info("%s: warm GPU failed, CPU fallback this turn", stage)
            return "cpu", cpu_fn()
        logger.info("%s won by gpu (warm shortcut) in %.2fs",
                    stage, time.perf_counter() - started)
        return "gpu", result

    started = time.perf_counter()
    gpu_fut = _race_pool.submit(_guard_gpu, stage, breaker, gpu_fn)
    cpu_fut = _race_pool.submit(cpu_fn)
    done, _ = wait([gpu_fut, cpu_fut], return_when=FIRST_COMPLETED)
    first = next(iter(done))
    try:
        winner, result = ("gpu" if first is gpu_fut else "cpu"), first.result()
    except Exception:
        other = cpu_fut if first is gpu_fut else gpu_fut
        # Blocks only until the remaining leg finishes; raises only if both failed.
        winner, result = ("cpu" if first is gpu_fut else "gpu"), other.result()
    # Warm = GPU won *quickly*; a slow GPU win means the worker is still cold.
    _gpu_warm[stage] = (winner == "gpu"
                        and time.perf_counter() - started < WARM_TIMEOUT_S)
    return winner, result


class RoutedSTTHandler(FasterWhisperSTTHandler):
    """Parakeet-on-RunPod raced with local faster-whisper; first finisher wins."""

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        parent_process = super().process
        if vad_audio.mode == "progressive" or not _stt_breaker.allow():
            yield from parent_process(vad_audio)
            return
        started = time.perf_counter()
        wav_b64 = _wav_b64(vad_audio.audio)

        def _gpu() -> list:
            out = _runsync(STT_ENDPOINT_ID, {"audio": wav_b64})
            return [Transcription(
                text=out["text"],
                language_code=None,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )]

        def _cpu() -> list:
            return list(parent_process(vad_audio))

        winner, items = _race("stt", _stt_breaker, _gpu, _cpu)
        logger.info("STT won by %s in %.2fs", winner, time.perf_counter() - started)
        yield from items


# --- Supertonic 3 CPU TTS (replaces Kokoro as the CPU fallback leg) ---
# 99M params, ONNX Runtime + INT8, ~2-3x realtime on 2-core ARM (vs Kokoro ~1x).
# Built-in voices M1-M5/F1-F5; custom cloned voices via SUPERTONIC_STYLE_PATH
# (Voice Builder JSON export). First run downloads ~400MB to ~/.cache/supertonic3/.
_supertonic_lock = threading.Lock()
_supertonic_tts = None
_supertonic_style = None


def _get_supertonic():
    """Lazy-load Supertonic 3 engine + voice style (thread-safe, once)."""
    global _supertonic_tts, _supertonic_style
    with _supertonic_lock:
        if _supertonic_tts is None:
            from supertonic import TTS
            _supertonic_tts = TTS(auto_download=True)
            style_path = os.environ.get("SUPERTONIC_STYLE_PATH")
            if style_path:
                _supertonic_style = _supertonic_tts.get_voice_style_from_path(style_path)
                logger.info("Supertonic 3 ready (custom style: %s)", style_path)
            else:
                voice = os.environ.get("SUPERTONIC_VOICE", "F1")
                _supertonic_style = _supertonic_tts.get_voice_style(voice_name=voice)
                logger.info("Supertonic 3 ready (built-in voice: %s)", voice)
    return _supertonic_tts, _supertonic_style


def _supertonic_synthesize(text: str) -> list:
    """Synthesize text with Supertonic 3 on CPU.

    Returns list of int16 numpy chunks at 16kHz (TTS_CHUNK_SAMPLES each),
    matching the pipeline's expected audio format.
    """
    from scipy.signal import resample_poly
    from math import gcd
    tts, style = _get_supertonic()
    steps = int(os.environ.get("SUPERTONIC_STEPS", "6"))
    wav, _duration = tts.synthesize(
        text=text,
        voice_style=style,
        total_steps=steps,
        lang="en",
    )
    # wav: float32 (1, N) at 44100 Hz -> 16kHz int16 chunks
    audio = wav.squeeze().astype(np.float32)
    g = gcd(44100, 16000)
    audio = resample_poly(audio, 16000 // g, 44100 // g).astype(np.float32)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return [pcm16[i:i + TTS_CHUNK_SAMPLES]
            for i in range(0, len(pcm16), TTS_CHUNK_SAMPLES)]


class RoutedTTSHandler(KokoroTTSHandler):
    """Qwen3-TTS-on-RunPod raced with local Supertonic 3; first finisher wins."""

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        parent_process = super().process
        if isinstance(tts_input, EndOfResponse):
            yield from parent_process(tts_input)
            return
        # Mirror the parent's speculative-turn guard so a cancelled turn never
        # burns GPU time.
        speculative_turns = getattr(self, "speculative_turns", None)
        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id, tts_input.turn_revision
        ):
            logger.debug("Dropping stale routed-TTS input for turn=%s rev=%s",
                         tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        if not _tts_breaker.allow():
            yield from parent_process(tts_input)
            return

        # Warm shortcut: stream sentence-by-sentence from the GPU only.
        # First sound lands after the first sentence (~1s) instead of after
        # the full reply (~3s). No CPU leg, no background burn.
        if _is_warm("tts"):
            _last_race_ts["tts"] = time.monotonic()
            lang = CODE_TO_QWEN3_LANG.get((tts_input.language_code or "en").lower(), "english")
            sentences = _split_sentences(tts_input.text)
            completed = 0
            gpu_ok = True
            try:
                for sentence in sentences:
                    out = _runsync(TTS_ENDPOINT_ID, {"text": sentence, "language": lang})
                    pcm16 = _wav_b64_to_int16(out["audio"])
                    if len(pcm16) == 0:
                        raise RuntimeError("GPU TTS returned empty audio for a sentence")
                    logger.info("tts: GPU streamed sentence %d/%d (%d chars) -> %.2fs audio",
                                completed + 1, len(sentences), len(sentence), len(pcm16) / 16000)
                    for i in range(0, len(pcm16), TTS_CHUNK_SAMPLES):
                        yield pcm16[i:i + TTS_CHUNK_SAMPLES]
                    completed += 1
            except requests.Timeout:
                logger.info("tts: streaming GPU leg timed out (cold worker?)")
                gpu_ok = False
            except Exception:
                _tts_breaker.failure()
                logger.warning("tts: streaming GPU leg errored", exc_info=True)
                gpu_ok = False
            else:
                _tts_breaker.success()
            _gpu_warm["tts"] = gpu_ok
            if not gpu_ok:
                # Resume only the unspoken remainder on CPU -- never duplicate
                # audio already streamed.
                remaining = " ".join(sentences[completed:])
                for chunk in _supertonic_synthesize(remaining):
                    yield chunk
            return

        started = time.perf_counter()
        lang = CODE_TO_QWEN3_LANG.get((tts_input.language_code or "en").lower(), "english")

        def _gpu() -> list:
            out = _runsync(TTS_ENDPOINT_ID, {"text": tts_input.text, "language": lang})
            pcm16 = _wav_b64_to_int16(out["audio"])
            logger.info("tts: GPU leg returned %.2fs of audio", len(pcm16) / 16000)
            chunks = [pcm16[i:i + TTS_CHUNK_SAMPLES]
                      for i in range(0, len(pcm16), TTS_CHUNK_SAMPLES)]
            if not chunks:
                # Winning with silence is worse than losing: force CPU fallback.
                raise RuntimeError("GPU TTS returned empty audio")
            return chunks

        def _cpu() -> list:
            # Supertonic 3 on CPU (~2-3x realtime) replaces Kokoro (~1x).
            # Voice via SUPERTONIC_VOICE (default F1) or SUPERTONIC_STYLE_PATH
            # (Voice Builder JSON for cloned voices, incl. Phase 2 profiles).
            return _supertonic_synthesize(tts_input.text)

        winner, chunks = _race("tts", _tts_breaker, _gpu, _cpu)
        logger.info("TTS won by %s in %.2fs", winner, time.perf_counter() - started)
        yield from chunks


def register_routed_backends() -> None:
    """Add 'runpod-routed' to the STT/TTS backend registries. Call before arg parsing."""
    from speech_to_speech import backend_registry
    from speech_to_speech.arguments_classes.faster_whisper_stt_arguments import (
        FasterWhisperSTTHandlerArguments,
    )
    from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
    from speech_to_speech.backend_registry import BackendSpec

    def _create_routed_stt(context, config):
        handler = RoutedSTTHandler(
            context.stop_event,
            queue_in=context.queue_in,
            queue_out=context.queue_out,
            setup_kwargs=dict(config),
        )
        handler.speculative_turns = context.speculative_turns
        return handler

    def _create_routed_tts(context, config):
        return RoutedTTSHandler(
            context.stop_event,
            queue_in=context.queue_in,
            queue_out=context.queue_out,
            setup_args=(context.should_listen,),
            setup_kwargs={
                **config,
                "cancel_scope": context.cancel_scope,
                "speculative_turns": context.speculative_turns,
            },
        )

    backend_registry.STT_BACKENDS["runpod-routed"] = BackendSpec(
        "runpod-routed",
        "stt",
        FasterWhisperSTTHandlerArguments,
        _create_routed_stt,
        config_prefix="faster_whisper_stt",
    )
    backend_registry.TTS_BACKENDS["runpod-routed"] = BackendSpec(
        "runpod-routed",
        "tts",
        KokoroTTSHandlerArguments,
        _create_routed_tts,
        config_prefix="kokoro",
    )
    logger.info("Registered backends: --stt runpod-routed, --tts runpod-routed")
