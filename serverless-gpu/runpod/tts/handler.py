"""RunPod serverless handler: Qwen3-TTS 1.7B CustomVoice TTS.

Same model + ggml backend as the Colab GPU rig (upstream Qwen3TTSHandler
uses faster-qwen3-tts on CUDA). Per-request synthesis: text in, one 16kHz
mono wav out. The Oracle router re-chunks it for streaming playback.

Input:  {"text": "...", "language": "english", "speaker": "Aiden",
         "instruct": null, "max_new_tokens": null}
Output: {"audio": "<base64 wav bytes, 16kHz mono PCM16>", "sample_rate": 16000}

Env:
  QWEN3_TTS_MODEL    HF repo id (default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
  QWEN3_TTS_SPEAKER  default preset speaker (default: Aiden)
  QWEN3_TTS_LANGUAGE default language name (default: english)
  HF_TOKEN           Hugging Face token (exported on the endpoint)
"""

import base64
import io
import logging
import os
from math import gcd

import numpy as np
import runpod
import soundfile as sf
import torch
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3-tts")

MODEL_ID = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_SPEAKER = os.environ.get("QWEN3_TTS_SPEAKER", "Aiden")
DEFAULT_LANGUAGE = os.environ.get("QWEN3_TTS_LANGUAGE", "english")
NATIVE_SR = 24000
PIPELINE_SR = 16000

logger.info("Loading Qwen3-TTS model: %s", MODEL_ID)
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
    gguf_talker_path=None,
    gguf_codec_path=None,
    qwentts_ref_cache_dir=None,
)
native_sr = int(getattr(model, "sample_rate", NATIVE_SR))
logger.info("Qwen3-TTS ready (native sr=%d)", native_sr)


def _chunk_to_float32(chunk) -> np.ndarray:
    if isinstance(chunk, bytes):
        return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    arr = np.asarray(chunk)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 32768.0
    return arr.astype(np.float32)


def _synthesize(text: str, language: str, speaker: str, instruct, max_new_tokens) -> bytes:
    if max_new_tokens is None:
        max_new_tokens = int(min(1536, max(360, len(text) * 6)))
    parts = []
    stream = model.generate_custom_voice_streaming(
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
        chunk_size=8,
        max_new_tokens=max_new_tokens,
        non_streaming_mode=False,
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


def handler(job: dict) -> dict:
    inp = job["input"]
    text = inp["text"]
    wav = _synthesize(
        text=text,
        language=inp.get("language") or DEFAULT_LANGUAGE,
        speaker=inp.get("speaker") or DEFAULT_SPEAKER,
        instruct=inp.get("instruct"),
        max_new_tokens=inp.get("max_new_tokens"),
    )
    logger.info("Synthesized %d chars -> %.1fs audio", len(text), len(wav) / 32044)
    return {"audio": base64.b64encode(wav).decode("ascii"), "sample_rate": PIPELINE_SR}


runpod.serverless.start({"handler": handler})
