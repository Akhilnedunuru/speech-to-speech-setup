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
  - TTS streams sentence-by-sentence (winner reused, mid-stream fallback,
    incremental frame flow verified)
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

        async def gpu(text, context_id):
            return pcm

        async def slow_cpu(text):
            await asyncio.sleep(0.2)
            return b""

        tts._gpu_synthesize = gpu
        tts._cpu_synthesize = slow_cpu
        frames = run(collect(tts.run_tts("hello", "test-ctx")))
        # NOTE: TTSStartedFrame is pushed by base class _push_tts_frames, not
        # by run_tts -- so frames[0] here is the first audio chunk.
        assert isinstance(frames[-1], TTSStoppedFrame), frames
        audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio) == 4, f"expected 4 chunks, got {len(audio)}"
        assert b"".join(f.audio for f in audio) == pcm
        assert all(f.sample_rate == 16000 and f.num_channels == 1 for f in audio)
        assert tts._state.gpu_warm is True
        print("PASS: test_tts_gpu_wins_chunks_frames")
    finally:
        restore()


def test_tts_streams_sentences_in_order_winner_reused():
    """Multi-sentence text: race once on sentence 1, winner reused for rest."""
    restore = with_timeouts()
    try:
        tts = fresh_tts()
        calls = []

        async def gpu(text, context_id):
            calls.append(("gpu", text))
            return b"\xAA" * 640  # 1 chunk per sentence

        async def cpu(text):
            calls.append(("cpu", text))
            return b"\xBB" * 640

        tts._gpu_synthesize = gpu
        tts._cpu_synthesize = cpu
        frames = run(collect(tts.run_tts("First one. Second one! Third one?", "ctx")))
        audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio) == 3, f"expected 3 sentence-chunks, got {len(audio)}"
        # GPU won the first-sentence race; it synthesizes all 3 sentences.
        # CPU fires once as the race loser for sentence 1 (expected in a race),
        # but its audio is never used and it is never called for sentences 2-3.
        gpu_texts = [c[1] for c in calls if c[0] == "gpu"]
        cpu_texts = [c[1] for c in calls if c[0] == "cpu"]
        assert gpu_texts == ["First one.", "Second one!", "Third one?"], calls
        assert cpu_texts == ["First one."], calls
        assert b"".join(f.audio for f in audio) == b"\xAA" * 640 * 3
        print("PASS: test_tts_streams_sentences_in_order_winner_reused")
    finally:
        restore()


def test_tts_cpu_wins_first_sentence_uses_cpu_for_rest():
    """CPU wins the first-sentence race -> GPU never called again."""
    restore = with_timeouts()
    try:
        tts = fresh_tts()
        calls = []

        async def slow_gpu(text, context_id):
            calls.append(("gpu", text))
            await asyncio.sleep(5)  # timeout -> CPU wins race
            return b"\xAA" * 640

        async def cpu(text):
            calls.append(("cpu", text))
            return b"\xBB" * 640

        tts._gpu_synthesize = slow_gpu
        tts._cpu_synthesize = cpu
        frames = run(collect(tts.run_tts("Alpha. Beta.", "ctx")))
        audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio) == 2, f"expected 2 chunks, got {len(audio)}"
        # GPU was tried once (lost the race); 2nd sentence went straight to CPU.
        gpu_calls = [c for c in calls if c[0] == "gpu"]
        cpu_calls = [c for c in calls if c[0] == "cpu"]
        assert len(gpu_calls) == 1, calls
        assert len(cpu_calls) == 2, calls
        assert b"".join(f.audio for f in audio) == b"\xBB" * 640 * 2
        print("PASS: test_tts_cpu_wins_first_sentence_uses_cpu_for_rest")
    finally:
        restore()


def test_tts_gpu_stumbles_midstream_falls_back_to_cpu():
    """GPU wins sentence 1, fails on sentence 2 -> CPU takes over."""
    restore = with_timeouts()
    try:
        tts = fresh_tts()
        calls = []

        async def flaky_gpu(text, context_id):
            calls.append(("gpu", text))
            if "Second" in text:
                raise RuntimeError("gpu died mid-stream")
            return b"\xAA" * 640

        async def cpu(text):
            calls.append(("cpu", text))
            return b"\xBB" * 640

        tts._gpu_synthesize = flaky_gpu
        tts._cpu_synthesize = cpu
        frames = run(collect(tts.run_tts("First here. Second here.", "ctx")))
        audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio) == 2, f"expected 2 chunks, got {len(audio)}"
        # Sentence 1 from GPU, sentence 2 fell back to CPU.
        assert b"".join(f.audio for f in audio) == b"\xAA" * 640 + b"\xBB" * 640
        assert not any(isinstance(f, ErrorFrame) for f in frames), "no ErrorFrame expected"
        print("PASS: test_tts_gpu_stumbles_midstream_falls_back_to_cpu")
    finally:
        restore()


def test_tts_streams_before_full_synthesis():
    """KEY: first-sentence audio must flow before 2nd sentence finishes.

    This is what keeps Pipecat's 3s audio-context alive -- batching (waiting
    for all sentences) would time out.
    """
    restore = with_timeouts()
    try:
        tts = fresh_tts()
        events = []

        async def gpu(text, context_id):
            events.append(f"gpu_start:{text[:12]}")
            if "Second" in text:
                await asyncio.sleep(0.3)  # slow 2nd sentence
            events.append(f"gpu_done:{text[:12]}")
            return b"\x00\x01" * 320  # 640 bytes = 1 chunk

        async def cpu(text):
            events.append(f"cpu:{text[:12]}")
            return b"\x02\x03" * 320

        tts._gpu_synthesize = gpu
        tts._cpu_synthesize = cpu

        first_audio_seen = []
        violation = []

        async def drive():
            async for f in tts.run_tts("First sentence here. Second sentence here.", "ctx"):
                if isinstance(f, TTSAudioRawFrame) and not first_audio_seen:
                    first_audio_seen.append(True)
                    if any("gpu_start:Second" in e for e in events):
                        violation.append(True)

        run(drive())
        assert first_audio_seen, "no audio frames were yielded at all"
        assert not violation, (
            "BATCHING DETECTED: 2nd sentence synthesis started before "
            f"1st-sentence audio flowed. events={events}"
        )
        print("PASS: test_tts_streams_before_full_synthesis")
    finally:
        restore()


def test_split_sentences():
    from race import _split_sentences
    assert _split_sentences("") == []
    assert _split_sentences("Hello.") == ["Hello."]
    assert _split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    # Long sentence (>80 chars) splits on clause boundaries
    long_s = ("This is a deliberately long sentence designed to exceed eighty characters, "
              "with a clause here, and another clause at the very end.")
    assert len(long_s) > 80, len(long_s)
    parts = _split_sentences(long_s)
    assert len(parts) >= 2, parts
    assert all(len(p) <= 80 for p in parts), parts
    assert " ".join(parts) == long_s, parts
    # No trailing punctuation still works
    assert _split_sentences("no punctuation at all") == ["no punctuation at all"]
    print("PASS: test_split_sentences")


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
    test_tts_streams_sentences_in_order_winner_reused()
    test_tts_cpu_wins_first_sentence_uses_cpu_for_rest()
    test_tts_gpu_stumbles_midstream_falls_back_to_cpu()
    test_tts_streams_before_full_synthesis()
    test_split_sentences()
    test_both_legs_fail_yields_error_frame()
    test_race_turn_direct_breaker_open()
    print("\nAll race tests passed.")
