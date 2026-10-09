"""Tests for voice.py — no pipecat, no network. Filesystem mocked via tmp dirs."""

import base64
import os
import tempfile

from voice import VoiceResolver


def make_profile(profiles_dir, profile_id, wav_bytes=b"FAKEWAV", text="hello world"):
    pdir = os.path.join(profiles_dir, profile_id)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, "reference.wav"), "wb") as f:
        f.write(wav_bytes)
    with open(os.path.join(pdir, "ref_text.txt"), "w") as f:
        f.write(text)
    return pdir


def make_resolver(tmp, default_b64="REVGQg==", default_text="default voice"):
    profiles_dir = os.path.join(tmp, "voice-profiles")
    os.makedirs(profiles_dir, exist_ok=True)
    active_file = os.path.join(profiles_dir, ".active_voice")
    return VoiceResolver(
        profiles_dir=profiles_dir,
        active_voice_file=active_file,
        default_ref_audio_b64=default_b64,
        default_ref_text=default_text,
    ), profiles_dir, active_file


def test_no_selection_uses_default():
    with tempfile.TemporaryDirectory() as tmp:
        r, _, _ = make_resolver(tmp)
        b64, text, pid = r.resolve()
        assert (b64, text, pid) == ("REVGQg==", "default voice", None), (b64, text, pid)
    print("PASS: test_no_selection_uses_default")


def test_selection_switches_voice():
    with tempfile.TemporaryDirectory() as tmp:
        r, profiles_dir, active_file = make_resolver(tmp)
        make_profile(profiles_dir, "alice-1", wav_bytes=b"ALICEWAV", text="alice says hi")
        with open(active_file, "w") as f:
            f.write("alice-1")
        b64, text, pid = r.resolve()
        assert pid == "alice-1", pid
        assert text == "alice says hi", text
        assert base64.b64decode(b64) == b"ALICEWAV"
    print("PASS: test_selection_switches_voice")


def test_switch_mid_session_reloads():
    with tempfile.TemporaryDirectory() as tmp:
        r, profiles_dir, active_file = make_resolver(tmp)
        make_profile(profiles_dir, "alice-1", text="alice")
        make_profile(profiles_dir, "bob-2", text="bob")
        with open(active_file, "w") as f:
            f.write("alice-1")
        assert r.resolve()[2] == "alice-1"
        # Switch: bump mtime by rewriting the file.
        with open(active_file, "w") as f:
            f.write("bob-2")
        os.utime(active_file, (os.path.getatime(active_file) + 2,
                               os.path.getmtime(active_file) + 2))
        b64, text, pid = r.resolve()
        assert pid == "bob-2", pid
        assert text == "bob", text
    print("PASS: test_switch_mid_session_reloads")


def test_invalid_id_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        r, profiles_dir, active_file = make_resolver(tmp)
        with open(active_file, "w") as f:
            f.write("../../etc/passwd")
        b64, text, pid = r.resolve()
        assert (b64, text, pid) == ("REVGQg==", "default voice", None)
    print("PASS: test_invalid_id_falls_back")


def test_missing_profile_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        r, profiles_dir, active_file = make_resolver(tmp)
        with open(active_file, "w") as f:
            f.write("ghost-9")
        b64, text, pid = r.resolve()
        assert pid is None and text == "default voice"
    print("PASS: test_missing_profile_falls_back")


def test_empty_file_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        r, profiles_dir, active_file = make_resolver(tmp)
        with open(active_file, "w") as f:
            f.write("   \n")
        assert r.resolve()[2] is None
    print("PASS: test_empty_file_falls_back")


def test_cache_hit_no_reread():
    """Second resolve() with unchanged mtime must not re-read files."""
    with tempfile.TemporaryDirectory() as tmp:
        r, profiles_dir, active_file = make_resolver(tmp)
        make_profile(profiles_dir, "alice-1", text="v1")
        with open(active_file, "w") as f:
            f.write("alice-1")
        first = r.resolve()
        # Corrupt the profile on disk; cached value should still win.
        with open(os.path.join(profiles_dir, "alice-1", "ref_text.txt"), "w") as f:
            f.write("v2-corrupted")
        second = r.resolve()
        assert second[1] == "v1", second[1]
        assert first == second
    print("PASS: test_cache_hit_no_reread")


def test_active_profile_id_property():
    with tempfile.TemporaryDirectory() as tmp:
        r, profiles_dir, active_file = make_resolver(tmp)
        assert r.active_profile_id is None
        make_profile(profiles_dir, "carol-3", text="carol")
        with open(active_file, "w") as f:
            f.write("carol-3")
        assert r.active_profile_id == "carol-3"
    print("PASS: test_active_profile_id_property")


if __name__ == "__main__":
    test_no_selection_uses_default()
    test_selection_switches_voice()
    test_switch_mid_session_reloads()
    test_invalid_id_falls_back()
    test_missing_profile_falls_back()
    test_empty_file_falls_back()
    test_cache_hit_no_reread()
    test_active_profile_id_property()
    print("\nAll voice tests passed.")
