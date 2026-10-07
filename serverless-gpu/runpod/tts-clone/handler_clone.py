"""RunPod serverless handler: Qwen3-TTS 1.7B Base voice cloning (ICL mode).

Zero-shot voice cloning: pass reference audio + transcript per request.
No per-voice training or redeployment -- one endpoint serves unlimited voices.

Input:  {"text": "...", "ref_audio": "<base64 wav>", "ref_text": "...",
         "language": "english"}
Output: {"audio": "<base64 wav bytes, 16kHz mono PCM16>", "sample_rate": 16000}

Env:
  QWEN3_TTS_MODEL    HF repo id (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)
  QWEN3_TTS_LANGUAGE default language name (default: english)
  HF_TOKEN           Hugging Face token (exported on the endpoint)
"""

import base64
import io
import logging
import os
import tempfile
from math import gcd

import numpy as np
import runpod
import soundfile as sf
import torch
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3-tts-clone")

MODEL_ID = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DEFAULT_LANGUAGE = os.environ.get("QWEN3_TTS_LANGUAGE", "english")
NATIVE_SR = 24000
PIPELINE_SR = 16000

logger.info("Loading Qwen3-TTS Base model: %s", MODEL_ID)
try:
    import importlib.metadata as _md
    logger.info("faster-qwen3-tts version: %s", _md.version("faster-qwen3-tts"))
except Exception:
    pass
from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = FasterQwen3TTS.from_pretrained(
    MODEL_ID,
    device="cuda",
    dtype=dtype,
    attn_implementation="eager",
    backend="ggml",
    quant="BF16",
)
native_sr = int(getattr(model, "sample_rate", NATIVE_SR))
logger.info("Qwen3-TTS Base ready (native sr=%d)", native_sr)


def _chunk_to_float32(chunk) -> np.ndarray:
    if isinstance(chunk, bytes):
        return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    arr = np.asarray(chunk)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 32768.0
    return arr.astype(np.float32)


def _synthesize(text: str, ref_audio_b64: str, ref_text: str, language: str) -> bytes:
    # Decode ref audio to a temp wav file (API takes a path)
    ref_wav = base64.b64decode(ref_audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(ref_wav)
        ref_path = f.name
    try:
        parts = []
        stream = model.generate_voice_clone_streaming(
            text=text,
            language=language,
            ref_audio=ref_path,
            ref_text=ref_text,
            chunk_size=8,
        )
        for chunk in stream:
            # ggml backend yields (audio_chunk, sample_rate, info) tuples
            if isinstance(chunk, (tuple, list)):
                chunk = chunk[0]
            parts.append(_chunk_to_float32(chunk))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if native_sr != PIPELINE_SR:
            g = gcd(native_sr, PIPELINE_SR)
            audio = resample_poly(audio, PIPELINE_SR // g, native_sr // g).astype(np.float32)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, pcm16, PIPELINE_SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    finally:
        try:
            os.unlink(ref_path)
        except OSError:
            pass


def handler(job: dict) -> dict:
    inp = job["input"]
    text = inp["text"]
    wav = _synthesize(
        text=text,
        ref_audio_b64=inp["ref_audio"],
        ref_text=inp["ref_text"],
        language=inp.get("language") or DEFAULT_LANGUAGE,
    )
    logger.info("Cloned %d chars -> %.1fs audio", len(text), len(wav) / 32044)
    return {"audio": base64.b64encode(wav).decode("ascii"), "sample_rate": PIPELINE_SR}


runpod.serverless.start({"handler": handler})
