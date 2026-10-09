"""Mocked tests for the Pipecat spike custom processors.

Verifies:
  1. RunPodParakeetSTT.run_stt() builds the right RunPod payload and yields
     a TranscriptionFrame with the returned text.
  2. RunPodQwenTTS.run_tts() builds the right payload (text + ref_audio +
     ref_text + language) and yields TTSAudioRawFrames with PCM audio.

Run:  python -m pytest test_processors.py -v   (or: python test_processors.py)
No real network calls are made; `requests` is monkeypatched.
"""

import asyncio
import base64
import io
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import spike
from spike import RunPodParakeetSTT, RunPodQwenTTS
from pipecat.frames.frames import (
    ErrorFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)


def _make_wav_bytes(text_marker: str = "x", seconds: float = 0.1, sr: int = 16000) -> bytes:
    """Build a minimal valid WAV (16-bit mono PCM) in memory."""
    n = int(sr * seconds)
    pcm = struct.pack(f"<{n}h", *([0] * n))
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patch_requests(monkey_calls: dict, response_payload: dict):
    """Replace spike.requests.post/get with recorders returning response_payload."""
    import spike as spike_mod

    orig_post = spike_mod.requests.post
    orig_get = spike_mod.requests.get

    def fake_post(url, headers=None, json=None, timeout=None):
        monkey_calls["post"] = {"url": url, "headers": headers, "json": json}
        return _FakeResponse(response_payload)

    def fake_get(url, headers=None, timeout=None):
        monkey_calls["get"] = {"url": url}
        return _FakeResponse(response_payload)

    spike_mod.requests.post = fake_post
    spike_mod.requests.get = fake_get
    return orig_post, orig_get


def _restore_requests(orig_post, orig_get):
    import spike as spike_mod

    spike_mod.requests.post = orig_post
    spike_mod.requests.get = orig_get


async def _collect(agen):
    return [f async for f in agen]


def test_stt_payload_and_frame():
    calls = {}
    wav = _make_wav_bytes()
    orig_post, orig_get = _patch_requests(
        calls, {"status": "COMPLETED", "output": {"text": "hello world"}}
    )
    try:
        stt = RunPodParakeetSTT(api_key="KEY", endpoint_id="stt-ep")
        frames = asyncio.run(_collect(stt.run_stt(wav)))
    finally:
        _restore_requests(orig_post, orig_get)

    assert calls["post"]["url"] == "https://api.runpod.ai/v2/stt-ep/runsync", calls
    body = calls["post"]["json"]["input"]
    assert base64.b64decode(body["audio"]) == wav, "STT must send base64 WAV"
    assert calls["post"]["headers"]["Authorization"] == "Bearer KEY"

    tf = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert len(tf) == 1 and tf[0].text == "hello world", frames
    assert not any(isinstance(f, ErrorFrame) for f in frames)
    print("PASS: test_stt_payload_and_frame")


def test_stt_failure_yields_error_frame():
    calls = {}
    orig_post, orig_get = _patch_requests(
        calls, {"status": "FAILED", "error": "boom"}
    )
    try:
        stt = RunPodParakeetSTT(api_key="KEY", endpoint_id="stt-ep")
        frames = asyncio.run(_collect(stt.run_stt(_make_wav_bytes())))
    finally:
        _restore_requests(orig_post, orig_get)

    assert any(isinstance(f, ErrorFrame) for f in frames), frames
    print("PASS: test_stt_failure_yields_error_frame")


def test_tts_payload_and_frames():
    calls = {}
    wav = _make_wav_bytes(seconds=0.2)  # 0.2s of silence -> PCM frames out
    ref_b64 = base64.b64encode(_make_wav_bytes()).decode()
    orig_post, orig_get = _patch_requests(
        calls,
        {"status": "COMPLETED", "output": {"audio": base64.b64encode(wav).decode()}},
    )
    try:
        tts = RunPodQwenTTS(
            api_key="KEY",
            endpoint_id="tts-ep",
            ref_audio_b64=ref_b64,
            ref_text="reference transcript",
        )
        frames = asyncio.run(_collect(tts.run_tts("Hello there")))
    finally:
        _restore_requests(orig_post, orig_get)

    assert calls["post"]["url"] == "https://api.runpod.ai/v2/tts-ep/runsync", calls
    body = calls["post"]["json"]["input"]
    assert body["text"] == "Hello there", body
    assert body["ref_audio"] == ref_b64, "TTS must send reference audio"
    assert body["ref_text"] == "reference transcript", body
    assert body["language"] == "english", body  # handler expects full name, not "en"

    kinds = [type(f).__name__ for f in frames]
    assert "TTSStartedFrame" in kinds, kinds
    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames, "expected TTSAudioRawFrame chunks"
    assert all(f.sample_rate == 16000 and f.num_channels == 1 for f in audio_frames)
    assert "TTSStoppedFrame" in kinds, kinds
    print("PASS: test_tts_payload_and_frames")


def test_tts_in_queue_polls():
    """runsync returns IN_QUEUE first; spike should poll /status/ until COMPLETED."""
    import spike as spike_mod

    wav = _make_wav_bytes(seconds=0.05)
    ref_b64 = base64.b64encode(_make_wav_bytes()).decode()
    calls = {"gets": 0}
    orig_post = spike_mod.requests.post
    orig_get = spike_mod.requests.get

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["post_url"] = url
        return _FakeResponse({"status": "IN_QUEUE", "id": "job-123"})

    def fake_get(url, headers=None, timeout=None):
        calls["gets"] += 1
        assert url == "https://api.runpod.ai/v2/tts-ep/status/job-123", url
        return _FakeResponse(
            {"status": "COMPLETED", "output": {"audio": base64.b64encode(wav).decode()}}
        )

    spike_mod.requests.post = fake_post
    spike_mod.requests.get = fake_get
    # Speed up: patch sleep inside _runsync via time module is overkill;
    # the poll loop sleeps min(2, remaining) once — acceptable for a test.
    try:
        tts = RunPodQwenTTS(
            api_key="KEY", endpoint_id="tts-ep",
            ref_audio_b64=ref_b64, ref_text="ref",
        )
        frames = asyncio.run(_collect(tts.run_tts("hi")))
    finally:
        spike_mod.requests.post = orig_post
        spike_mod.requests.get = orig_get

    assert calls["gets"] >= 1, "should have polled the status endpoint"
    assert any(isinstance(f, TTSAudioRawFrame) for f in frames)
    print("PASS: test_tts_in_queue_polls")


if __name__ == "__main__":
    test_stt_payload_and_frame()
    test_stt_failure_yields_error_frame()
    test_tts_payload_and_frames()
    test_tts_in_queue_polls()
    print("\nAll spike processor tests passed.")
