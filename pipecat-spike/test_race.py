"""Mocked tests for race.py — no network, no GPU, no model downloads.

The GPU/CPU legs are monkeypatched with fast async fakes so every race
rule can be verified deterministically:
  - GPU wins -> marked warm
  - GPU timeout -> CPU wins, breaker NOT tripped
  - real GPU errors -> breaker trips after MAX_FAILURES, GPU skipped
  - warm shortcut -> CPU leg never fires
  - idle > 240s -> warm reset, full race again
  - warm GPU stumbles -> CPU fallback, warm cleared
  - TTS frames chunked correctly; both-legs-failed -> ErrorFrame
"""

import asyncio
import time

from pipecat.frames.frames import (
    ErrorFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

import race
from race import RacingSTT, RacingTTS, _RaceState, _race_turn


def run(coro):
    return asyncio.run(coro)


async def collect(agen):
    return [f async for f in agen]


def text_of(frames):
    for f in frames:
        if isinstance(f, TranscriptionFrame):
            return f.text
    return None


def fresh_stt():
    return RacingSTT(api_key="k", endpoint_id="e")


def fresh_tts():
    return RacingTTS(api_key="k", endpoint_id="e", ref_audio_b64="x", ref_text="y")


def with_timeouts(gpu=5.0, warm=5.0):
    """Temporarily shrink race timeouts for fast tests. Returns restore fn."""
    old = (race.GPU_TIMEOUT_S, race.WARM_TIMEOUT_S)
    race.GPU_TIMEOUT_S, race.WARM_TIMEOUT_S = gpu, warm

    def restore():
        race.GPU_TIMEOUT_S, race.WARM_TIMEOUT_S = old

    return restore


def test_stt_gpu_wins_and_marks_warm():
    restore = with_timeouts()
    try:
        stt = fresh_stt()

        async def gpu(audio):
            return "gpu text"

        async def cpu(audio):
            await asyncio.sleep(0.5)
            return "cpu text"

        stt._gpu_transcribe = gpu
        stt._cpu_transcribe = cpu
        frames = run(collect(stt.run_stt(b"fake-wav")))
        assert text_of(frames) == "gpu text", frames
        assert stt._state.gpu_warm is True, "GPU should be marked warm after quick win"
        print("PASS: test_stt_gpu_wins_and_marks_warm")
    finally:
        restore()


def test_stt_cpu_wins_on_gpu_timeout_breaker_untouched():
    restore = with_timeouts(gpu=0.05)
    try:
        stt = fresh_stt()

        async def gpu(audio):
            await asyncio.sleep(5)  # wait_for cancels at 0.05s -> cold
            return "gpu text"

        async def cpu(audio):
            return "cpu text"

        stt._gpu_transcribe = gpu
        stt._cpu_transcribe = cpu
        frames = run(collect(stt.run_stt(b"fake-wav")))
        assert text_of(frames) == "cpu text", frames
        assert stt._state.gpu_warm is False
        assert stt._state.breaker._failures == 0, "timeouts must not count as failures"
        assert stt._state.breaker.allow() is True
        print("PASS: test_stt_cpu_wins_on_gpu_timeout_breaker_untouched")
    finally:
        restore()


def test_breaker_trips_on_real_errors_and_skips_gpu():
    restore = with_timeouts()
    try:
        stt = fresh_stt()
        calls = {"gpu": 0}

        async def gpu(audio):
            calls["gpu"] += 1
            raise RuntimeError("boom")

        async def cpu(audio):
            return "cpu text"

        stt._gpu_transcribe = gpu
        stt._cpu_transcribe = cpu
        for _ in range(3):  # MAX_FAILURES
            frames = run(collect(stt.run_stt(b"fake-wav")))
            assert text_of(frames) == "cpu text"
        assert stt._state.breaker.allow() is False, "breaker should be open"

        calls["gpu"] = 0
        frames = run(collect(stt.run_stt(b"fake-wav")))
        assert text_of(frames) == "cpu text"
        assert calls["gpu"] == 0, "GPU leg must be skipped while breaker is open"
        print("PASS: test_breaker_trips_on_real_errors_and_skips_gpu")
    finally:
        restore()


def test_warm_shortcut_skips_cpu():
    restore = with_timeouts()
    try:
        stt = fresh_stt()

        async def gpu(audio):
            return "gpu text"

        async def slow_cpu(audio):
            await asyncio.sleep(0.2)
            return "cpu text"

        stt._gpu_transcribe = gpu
        stt._cpu_transcribe = slow_cpu
        run(collect(stt.run_stt(b"fake-wav")))
        assert stt._state.gpu_warm is True

        async def bad_cpu(audio):
            raise AssertionError("CPU leg must be skipped on warm shortcut")

        stt._cpu_transcribe = bad_cpu
        frames = run(collect(stt.run_stt(b"fake-wav")))
        assert text_of(frames) == "gpu text", frames
        print("PASS: test_warm_shortcut_skips_cpu")
    finally:
        restore()


def test_idle_reset_reruns_full_race():
    restore = with_timeouts()
    try:
        stt = fresh_stt()

        async def gpu(audio):
            return "gpu text"

        async def cpu(audio):
            return "cpu text"

        stt._gpu_transcribe = gpu
        stt._cpu_transcribe = cpu
        run(collect(stt.run_stt(b"fake-wav")))
        assert stt._state.gpu_warm is True

        # Simulate >240s idle: warm flag must reset and CPU must fire again.
        stt._state.last_race_ts = time.monotonic() - 300
        fired = {"cpu": False}

        async def tracking_cpu(audio):
            fired["cpu"] = True
            return "cpu text"

        stt._cpu_transcribe = tracking_cpu
        run(collect(stt.run_stt(b"fake-wav")))
        assert fired["cpu"] is True, "full race should run again after idle reset"
        print("PASS: test_idle_reset_reruns_full_race")
    finally:
        restore()


def test_warm_gpu_stumble_falls_back_and_clears_warm():
    restore = with_timeouts(warm=0.05)
    try:
        stt = fresh_stt()
        stt._state.gpu_warm = True
        stt._state.last_race_ts = time.monotonic()

        async def slow_gpu(audio):
            await asyncio.sleep(5)  # warm cap is 0.05s -> stumble
            return "gpu text"

        async def cpu(audio):
            return "cpu text"

        stt._gpu_transcribe = slow_gpu
        stt._cpu_transcribe = cpu
        frames = run(collect(stt.run_stt(b"fake-wav")))
        assert text_of(frames) == "cpu text", frames
        assert stt._state.gpu_warm is False, "warm flag must clear on stumble"
        print("PASS: test_warm_gpu_stumble_falls_back_and_clears_warm")
    finally:
        restore()


def test_tts_gpu_wins_chunks_frames():
    restore = with_timeouts()
    try:
        tts = fresh_tts()
        pcm = b"\x00\x01" * 1000  # 2000 bytes -> 4 chunks of 640/640/640/80

        async def gpu(text):
            return pcm

        async def slow_cpu(text):
            await asyncio.sleep(0.2)
            return b""

        tts._gpu_synthesize = gpu
        tts._cpu_synthesize = slow_cpu
        frames = run(collect(tts.run_tts("hello", "test-ctx")))
        assert isinstance(frames[0], TTSStartedFrame), frames
        assert isinstance(frames[-1], TTSStoppedFrame), frames
        audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio) == 4, f"expected 4 chunks, got {len(audio)}"
        assert b"".join(f.audio for f in audio) == pcm
        assert all(f.sample_rate == 16000 and f.num_channels == 1 for f in audio)
        assert tts._state.gpu_warm is True
        print("PASS: test_tts_gpu_wins_chunks_frames")
    finally:
        restore()


def test_both_legs_fail_yields_error_frame():
    restore = with_timeouts()
    try:
        stt = fresh_stt()

        async def gpu(audio):
            raise RuntimeError("gpu dead")

        async def cpu(audio):
            raise RuntimeError("cpu dead")

        stt._gpu_transcribe = gpu
        stt._cpu_transcribe = cpu
        frames = run(collect(stt.run_stt(b"fake-wav")))
        assert any(isinstance(f, ErrorFrame) for f in frames), frames
        print("PASS: test_both_legs_fail_yields_error_frame")
    finally:
        restore()


def test_race_turn_direct_breaker_open():
    async def main():
        state = _RaceState("stt")
        state.breaker._cooldown_until = time.monotonic() + 300  # force open
        called = {"gpu": False}

        async def gpu():
            called["gpu"] = True
            return "g"

        async def cpu():
            return "c"

        winner, result = await _race_turn(state, gpu, cpu)
        assert (winner, result) == ("cpu", "c")
        assert called["gpu"] is False

    run(main())
    print("PASS: test_race_turn_direct_breaker_open")


if __name__ == "__main__":
    test_stt_gpu_wins_and_marks_warm()
    test_stt_cpu_wins_on_gpu_timeout_breaker_untouched()
    test_breaker_trips_on_real_errors_and_skips_gpu()
    test_warm_shortcut_skips_cpu()
    test_idle_reset_reruns_full_race()
    test_warm_gpu_stumble_falls_back_and_clears_warm()
    test_tts_gpu_wins_chunks_frames()
    test_both_legs_fail_yields_error_frame()
    test_race_turn_direct_breaker_open()
    print("\nAll race tests passed.")
