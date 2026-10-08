"""Voice cloning web UI (Phase 2).

Two pages:
  /     - Clone a voice: record yourself reading a paragraph, auto-transcribe, save profile
  /try  - Try it: pick a saved voice, type text, hear it in that voice

Design: the UI calls the RunPod Qwen3-TTS clone endpoint DIRECTLY.
No GPU/CPU race, no Supertonic fallback. If the endpoint is cold, the
request simply waits (~60s) and the frontend shows a "warming up" spinner.
Consistency over speed -- the user always hears their own voice.

Profiles live in ~/voice-profiles/<profile-id>/ as:
  reference.wav  - 16kHz mono WAV of the voice sample
  ref_text.txt   - exact transcript of reference.wav
  meta.json      - {name, created_at}

Env vars:
  RUNPOD_API_KEY              RunPod API key
  RUNPOD_STT_ENDPOINT_ID      Parakeet STT endpoint (default: acuml4hbia4g1f)
  RUNPOD_TTS_CLONE_ENDPOINT_ID  Qwen3-TTS clone endpoint (default: naq5rfqu0i3g7m)
  VOICE_UI_PORT               Port to listen on (default: 8080)
"""

import base64
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from functools import wraps

import requests
from flask import Flask, jsonify, render_template, request, send_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-ui")

app = Flask(__name__)
# 10MB upload cap (a 20s 16kHz mono WAV is ~640KB; generous headroom)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
STT_ENDPOINT_ID = os.environ.get("RUNPOD_STT_ENDPOINT_ID", "acuml4hbia4g1f")
CLONE_ENDPOINT_ID = os.environ.get("RUNPOD_TTS_CLONE_ENDPOINT_ID", "naq5rfqu0i3g7m")
PROFILES_DIR = os.path.expanduser("~/voice-profiles")

# Synthesis guardrails
MAX_SYNTH_CHARS = 500          # per-request text cap
RUNPOD_TIMEOUT_S = 180         # covers a full cold start + synthesis

# ---- Simple in-memory rate limiter: 10 req/min per IP ----
_RATE_WINDOW_S = 60
_RATE_MAX = 10
_rate_buckets: dict[str, list[float]] = {}


def rate_limited(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
        now = time.time()
        bucket = _rate_buckets.setdefault(ip, [])
        bucket[:] = [t for t in bucket if now - t < _RATE_WINDOW_S]
        if len(bucket) >= _RATE_MAX:
            return jsonify({"error": "Rate limit exceeded (10 req/min). Please wait a moment."}), 429
        bucket.append(now)
        return fn(*args, **kwargs)
    return wrapper


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:32] or "voice"


def _profile_path(profile_id: str) -> str:
    # Guard against path traversal: only allow safe slugs
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", profile_id or ""):
        raise ValueError("Invalid profile id")
    return os.path.join(PROFILES_DIR, profile_id)


def _runsync(endpoint_id: str, payload: dict) -> dict:
    """POST to a RunPod serverless endpoint via /runsync.

    When the endpoint is cold or busy, RunPod returns
    {"status": "IN_QUEUE", "id": "<job-id>"} instead of blocking until the
    job completes. In that case we poll the job status endpoint every
    2 seconds until the job completes or fails, bounded by RUNPOD_TIMEOUT_S.
    """
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    deadline = time.time() + RUNPOD_TIMEOUT_S

    def _check_output(data: dict) -> dict:
        output = data.get("output")
        if not isinstance(output, dict):
            raise RuntimeError(f"Unexpected RunPod output: {output!r}"[:200])
        return output

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
        raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown error')}"[:200])
    if status not in ("IN_QUEUE", "IN_PROGRESS"):
        raise RuntimeError(f"RunPod job did not complete: {status}")

    job_id = data.get("id")
    if not job_id:
        raise RuntimeError(f"RunPod job queued but no job id returned: {data!r}"[:200])

    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise RuntimeError(
                f"RunPod job {job_id} timed out after {RUNPOD_TIMEOUT_S}s"
            )
        time.sleep(min(2, remaining))
        poll_resp = requests.get(
            status_url, headers=headers, timeout=max(1, min(30, remaining))
        )
        poll_resp.raise_for_status()
        data = poll_resp.json()
        status = data.get("status")
        if status == "COMPLETED":
            return _check_output(data)
        if status == "FAILED":
            raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown error')}"[:200])
        # IN_QUEUE / IN_PROGRESS / anything else: keep polling until deadline.


# ---- Pages ----

@app.get("/")
def clone_page():
    return render_template("clone.html")


@app.get("/try")
def try_page():
    return render_template("try.html")


@app.get("/health")
def health():
    return jsonify({"ok": True})


# ---- API ----

@app.get("/api/profiles")
@rate_limited
def list_profiles():
    profiles = []
    if os.path.isdir(PROFILES_DIR):
        for pid in sorted(os.listdir(PROFILES_DIR)):
            meta_path = os.path.join(PROFILES_DIR, pid, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                profiles.append({
                    "id": pid,
                    "name": meta.get("name", pid),
                    "created_at": meta.get("created_at", ""),
                })
            except (OSError, json.JSONDecodeError):
                continue
    return jsonify({"profiles": profiles})


@app.post("/api/profile")
@rate_limited
def create_profile():
    """Accept a WAV recording + voice name, auto-transcribe it, save a profile."""
    if not RUNPOD_API_KEY:
        return jsonify({"error": "Server is missing RUNPOD_API_KEY."}), 500

    name = (request.form.get("name") or "").strip()
    consent = request.form.get("consent") == "true"
    audio_file = request.files.get("audio")

    if not name:
        return jsonify({"error": "Please give your voice a name."}), 400
    if not consent:
        return jsonify({"error": "Please confirm the consent checkbox."}), 400
    if audio_file is None:
        return jsonify({"error": "No audio received. Please record again."}), 400

    wav_bytes = audio_file.read()
    if len(wav_bytes) < 10_000:
        return jsonify({"error": "Recording too short. Please read the full paragraph."}), 400

    # Auto-transcribe via the Parakeet STT endpoint
    try:
        logger.info("Transcribing %.1fKB reference audio...", len(wav_bytes) / 1024)
        out = _runsync(STT_ENDPOINT_ID, {"audio": base64.b64encode(wav_bytes).decode()})
        transcript = (out.get("text") or "").strip()
    except Exception as e:
        logger.exception("STT transcription failed")
        return jsonify({"error": f"Transcription failed: {e}"}), 502

    if not transcript:
        return jsonify({"error": "Could not transcribe the recording. Please try again, speaking clearly."}), 502

    # Save profile: slug + timestamp suffix to avoid collisions
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    profile_id = f"{_slugify(name)}-{stamp}"
    pdir = _profile_path(profile_id)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, "reference.wav"), "wb") as f:
        f.write(wav_bytes)
    with open(os.path.join(pdir, "ref_text.txt"), "w") as f:
        f.write(transcript)
    with open(os.path.join(pdir, "meta.json"), "w") as f:
        json.dump({
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "transcript_chars": len(transcript),
        }, f, indent=2)

    logger.info("Saved profile %s (%d chars)", profile_id, len(transcript))
    return jsonify({
        "profile_id": profile_id,
        "name": name,
        "transcript": transcript,
    })


@app.post("/api/synthesize")
@rate_limited
def synthesize():
    """Synthesize text in a saved voice. Calls the clone endpoint directly."""
    if not RUNPOD_API_KEY:
        return jsonify({"error": "Server is missing RUNPOD_API_KEY."}), 500

    data = request.get_json(force=True, silent=True) or {}
    profile_id = (data.get("profile_id") or "").strip()
    text = (data.get("text") or "").strip()

    if not profile_id or not text:
        return jsonify({"error": "profile_id and text are required."}), 400
    if len(text) > MAX_SYNTH_CHARS:
        return jsonify({"error": f"Text too long (max {MAX_SYNTH_CHARS} chars)."}), 400

    try:
        pdir = _profile_path(profile_id)
    except ValueError:
        return jsonify({"error": "Unknown voice."}), 404
    ref_wav = os.path.join(pdir, "reference.wav")
    ref_txt = os.path.join(pdir, "ref_text.txt")
    if not (os.path.isfile(ref_wav) and os.path.isfile(ref_txt)):
        return jsonify({"error": "Unknown voice."}), 404

    with open(ref_wav, "rb") as f:
        ref_audio_b64 = base64.b64encode(f.read()).decode()
    with open(ref_txt) as f:
        ref_text = f.read().strip()

    try:
        logger.info("Synthesizing %d chars with profile %s...", len(text), profile_id)
        out = _runsync(CLONE_ENDPOINT_ID, {
            "text": text,
            "ref_audio": ref_audio_b64,
            "ref_text": ref_text,
            "language": "en",
        })
        audio_b64 = out.get("audio")
        if not audio_b64:
            raise RuntimeError("Clone endpoint returned no audio.")
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        logger.exception("Synthesis failed")
        return jsonify({"error": f"Synthesis failed: {e}"}), 502

    return send_file(
        io.BytesIO(audio_bytes),
        mimetype="audio/wav",
        as_attachment=False,
        download_name="cloned.wav",
    )


if __name__ == "__main__":
    os.makedirs(PROFILES_DIR, exist_ok=True)
    port = int(os.environ.get("VOICE_UI_PORT", "8080"))
    if not RUNPOD_API_KEY:
        logger.warning("RUNPOD_API_KEY is not set -- /api/* calls will fail until it is exported.")
    logger.info("Voice UI listening on port %d (profiles: %s)", port, PROFILES_DIR)
    app.run(host="0.0.0.0", port=port, threaded=True)
