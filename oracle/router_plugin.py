"""
GPU-routed STT/TTS backends for the Oracle 24/7 box.

Registers two new backends with the speech-to-speech backend registry:

    --stt runpod-routed    Parakeet TDT on RunPod serverless -> faster-whisper CPU fallback
    --tts runpod-routed    Qwen3-TTS on RunPod serverless   -> Kokoro CPU fallback

Per-turn behavior:
  1. If the circuit breaker allows, POST to the RunPod endpoint (runsync)
     with a short client-side timeout (GPU_ROUTE_TIMEOUT_S, default 8s).
  2. On a cold start the server keeps booting the worker after our client
     times out; this turn is answered from CPU, the NEXT turns hit the warm GPU.
  3. Consecutive failures trip the breaker: the GPU leg is skipped for
     GPU_ROUTE_COOLDOWN_S (default 300s) so a dead endpoint adds zero latency.

The router subclasses the stock CPU handlers, so setup(), config flags and
the queue/thread contract are inherited unchanged. Only process() is overridden.

Env:
  RUNPOD_API_KEY            RunPod API key (required)
  RUNPOD_STT_ENDPOINT_ID    serverless endpoint id for STT (required for --stt runpod-routed)
  RUNPOD_TTS_ENDPOINT_ID    serverless endpoint id for TTS (required for --tts runpod-routed)
  GPU_ROUTE_TIMEOUT_S       per-request timeout, seconds (default 8)
  GPU_ROUTE_MAX_FAILURES    failures before cooldown (default 3)
  GPU_ROUTE_COOLDOWN_S      cooldown after tripping, seconds (default 300)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Iterator

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

# Whisper-style language code -> Qwen3-TTS language name (mirrors upstream aliases)
CODE_TO_QWEN3_LANG = {
    "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean",
    "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese",
    "es": "spanish", "it": "italian", "hi": "hindi",
}

TTS_CHUNK_SAMPLES = 1600  # 100 ms @ 16 kHz


class CircuitBreaker:
    """Skip the GPU leg for COOLDOWN_S after MAX_FAILURES consecutive failures."""

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
                "%s: circuit open for %.0fs after %d consecutive GPU failures",
                self.name, COOLDOWN_S, MAX_FAILURES,
            )


_stt_breaker = CircuitBreaker("runpod-stt")
_tts_breaker = CircuitBreaker("runpod-tts")


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


class RoutedSTTHandler(FasterWhisperSTTHandler):
    """Try Parakeet-on-RunPod first; fall back to local faster-whisper."""

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode == "progressive" or not _stt_breaker.allow():
            yield from super().process(vad_audio)
            return
        started = time.perf_counter()
        try:
            out = _runsync(STT_ENDPOINT_ID, {"audio": _wav_b64(vad_audio.audio)})
            _stt_breaker.success()
            logger.info("STT via RunPod GPU in %.2fs", time.perf_counter() - started)
            yield Transcription(
                text=out["text"],
                language_code=None,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.speech_end_at_s,
            )
            return
        except Exception as exc:
            _stt_breaker.failure()
            logger.warning("RunPod STT failed (%s); CPU fallback", exc)
        yield from super().process(vad_audio)


class RoutedTTSHandler(KokoroTTSHandler):
    """Try Qwen3-TTS-on-RunPod first; fall back to local Kokoro."""

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            yield from super().process(tts_input)
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

        if _tts_breaker.allow():
            started = time.perf_counter()
            try:
                lang = CODE_TO_QWEN3_LANG.get((tts_input.tts_language_code or "en").lower(), "english")
                out = _runsync(TTS_ENDPOINT_ID, {"text": tts_input.text, "language": lang})
                _tts_breaker.success()
                logger.info("TTS via RunPod GPU in %.2fs", time.perf_counter() - started)
                pcm16 = _wav_b64_to_int16(out["audio"])
                for i in range(0, len(pcm16), TTS_CHUNK_SAMPLES):
                    yield pcm16[i:i + TTS_CHUNK_SAMPLES]
                return
            except Exception as exc:
                _tts_breaker.failure()
                logger.warning("RunPod TTS failed (%s); CPU fallback", exc)
        yield from super().process(tts_input)


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
