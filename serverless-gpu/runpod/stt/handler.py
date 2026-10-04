"""RunPod serverless handler: Parakeet TDT 0.6B STT.

Same model + inference path as the Colab GPU rig (upstream
ParakeetTDTSTTHandler uses nano-parakeet on CUDA): float32 16kHz mono
audio in, transcription text out. Stateless per request.

Input:  {"audio": "<base64 wav bytes>"}
Output: {"text": "<transcription>"}

Env:
  PARAKEET_MODEL  HF repo id (default: nvidia/parakeet-tdt-0.6b-v3)
  HF_TOKEN        Hugging Face token (exported on the endpoint)
"""

import base64
import io
import logging
import os
from math import gcd

import numpy as np
import runpod
import soundfile as sf
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parakeet-stt")

MODEL_ID = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")

logger.info("Loading Parakeet model: %s", MODEL_ID)
from nano_parakeet import from_pretrained  # noqa: E402

import torch  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
model = from_pretrained(model_name=MODEL_ID, device=device)
model.transcribe(np.zeros(16000, dtype=np.float32))  # warmup
logger.info("Parakeet ready on %s", device)


def _to_mono_16k(raw_wav: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(raw_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        audio = resample_poly(audio, 16000 // g, int(sr) // g).astype(np.float32)
    return audio.astype(np.float32)


def handler(job: dict) -> dict:
    audio_b64 = job["input"]["audio"]
    audio = _to_mono_16k(base64.b64decode(audio_b64))
    text = model.transcribe(audio).strip()
    logger.info("Transcribed %.1fs -> %d chars", len(audio) / 16000, len(text))
    return {"text": text}


runpod.serverless.start({"handler": handler})
