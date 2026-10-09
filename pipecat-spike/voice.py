"""Phase 3, step 3: dynamic voice resolution for Pipecat TTS.

Ports the voice-selection logic from oracle/router_plugin.py
(_resolve_profile_dir / _get_clone_profile) into a reusable class.

The web UI's "Use for agent" button writes a profile ID to
~/voice-profiles/.active_voice. VoiceResolver checks that file's mtime on
each call and reloads the profile when the selection changes, so the agent's
voice can switch mid-session with no restart.

Check order:
  1. ~/voice-profiles/.active_voice — if it names a valid profile (safe
     slug + reference.wav + ref_text.txt present), use it.
  2. Fallback: constructor-provided default (ref_audio_b64, ref_text), which
     the caller loads from REF_AUDIO_PATH / REF_TEXT_PATH env vars.

The CPU fallback leg (Supertonic 3) is NOT covered: it uses a fixed
built-in voice (SUPERTONIC_VOICE, default F1) or a Voice Builder JSON
(SUPERTONIC_STYLE_PATH). Only the GPU Qwen clone leg switches voices,
because Qwen does zero-shot ICL per request while Supertonic needs a
pre-built style. Documented limitation, same as production.

No pipecat dependency: pure filesystem + base64 logic, fully unit-testable.
"""

import base64
import logging
import os
import re
import threading

logger = logging.getLogger("pipecat-voice")

_PROFILES_DIR = os.path.expanduser("~/voice-profiles")
_ACTIVE_VOICE_FILE = os.path.join(_PROFILES_DIR, ".active_voice")
_PROFILE_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


class VoiceResolver:
    """Resolves which voice the TTS GPU leg should clone, per turn.

    mtime-cached: reference files are only re-read when the .active_voice
    selection actually changes. Thread-safe.
    """

    def __init__(
        self,
        profiles_dir: str | None = None,
        active_voice_file: str | None = None,
        default_ref_audio_b64: str = "",
        default_ref_text: str = "",
    ):
        self._profiles_dir = profiles_dir or _PROFILES_DIR
        self._active_voice_file = active_voice_file or _ACTIVE_VOICE_FILE
        self._default = (default_ref_audio_b64, default_ref_text)
        self._lock = threading.Lock()
        self._cache_key = None  # (active_mtime or None, profile_dir or None)
        self._cache_value = None  # (ref_audio_b64, ref_text, profile_id or None)

    def resolve(self) -> tuple[str, str, str | None]:
        """Return (ref_audio_b64, ref_text, profile_id).

        profile_id is the UI-selected ID, or None when using the default.
        """
        with self._lock:
            profile_dir, mtime, pid = self._resolve_profile_dir()
            key = (mtime, profile_dir)
            if self._cache_value is None or self._cache_key != key:
                first_load = self._cache_key is None
                self._cache_key = key
                if not first_load:
                    if pid:
                        logger.info("Agent voice switched to: %s", pid)
                    else:
                        logger.info("Agent voice reverted to default profile")
                self._cache_value = self._load(profile_dir, pid)
            return self._cache_value

    @property
    def active_profile_id(self) -> str | None:
        """The currently selected profile ID, or None for default."""
        return self.resolve()[2]

    def _resolve_profile_dir(self) -> tuple[str | None, float | None, str | None]:
        """Pick the profile dir: UI-selected voice or default.

        Returns (profile_dir_or_None, active_mtime_or_None, profile_id_or_None).
        Falls back to default (with a warning) when .active_voice is missing,
        empty, invalid, or points at a profile without reference files.
        Mirrors oracle/router_plugin._resolve_profile_dir.
        """
        try:
            with open(self._active_voice_file) as f:
                profile_id = f.read().strip()
        except FileNotFoundError:
            return None, None, None
        except OSError as e:
            logger.warning(
                "Could not read %s: %s; using default voice",
                self._active_voice_file, e,
            )
            return None, None, None
        if not profile_id:
            return None, None, None
        if not _PROFILE_ID_RE.fullmatch(profile_id):
            logger.warning(
                "Invalid agent voice id %r in %s; using default voice",
                profile_id, self._active_voice_file,
            )
            return None, None, None
        candidate = os.path.join(self._profiles_dir, profile_id)
        if not (
            os.path.isfile(os.path.join(candidate, "reference.wav"))
            and os.path.isfile(os.path.join(candidate, "ref_text.txt"))
        ):
            logger.warning(
                "Agent voice profile '%s' missing reference files; using default voice",
                profile_id,
            )
            return None, None, None
        return candidate, os.path.getmtime(self._active_voice_file), profile_id

    def _load(
        self, profile_dir: str | None, profile_id: str | None
    ) -> tuple[str, str, str | None]:
        """Load (ref_audio_b64, ref_text, profile_id) from dir or default."""
        if profile_dir:
            ref_wav = os.path.join(profile_dir, "reference.wav")
            ref_txt = os.path.join(profile_dir, "ref_text.txt")
            with open(ref_wav, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("ascii")
            with open(ref_txt) as f:
                text = f.read().strip()
            logger.info(
                "Voice clone profile loaded: %s (%d chars transcript)",
                profile_dir, len(text),
            )
            return audio_b64, text, profile_id
        return self._default[0], self._default[1], None


def load_default_voice_from_env() -> tuple[str, str]:
    """Load (base64_wav, text) from REF_AUDIO_PATH / REF_TEXT_PATH.

    Empty strings when unset — the caller decides whether that's an error.
    """
    ref_audio_path = os.environ.get("REF_AUDIO_PATH", "")
    ref_text_path = os.environ.get("REF_TEXT_PATH", "")
    if not ref_audio_path or not ref_text_path:
        return "", ""
    with open(ref_audio_path, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode("ascii")
    with open(ref_text_path) as f:
        ref_text = f.read().strip()
    return ref_b64, ref_text
