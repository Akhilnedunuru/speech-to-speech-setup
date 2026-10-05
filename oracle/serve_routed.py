#!/usr/bin/env python3
"""Launch `speech-to-speech serve` with the runpod-routed STT/TTS backends.

Usage:
    python serve_routed.py --host 0.0.0.0 --port 8765 \\
        --stt runpod-routed --tts runpod-routed \\
        --faster_whisper_stt_model_name small.en \\
        --llm_backend responses-api --model_name "openai/gpt-oss-20b:groq" \\
        --responses_api_base_url "https://router.huggingface.co/v1" \\
        --responses_api_api_key "$HF_TOKEN" --responses_api_stream

All stock `serve` flags keep working; --stt/--tts additionally accept
"runpod-routed". Mix freely, e.g. --stt runpod-routed --tts kokoro.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router_plugin import register_routed_backends

register_routed_backends()

from speech_to_speech.s2s_pipeline import run_pipeline_command

if __name__ == "__main__":
    run_pipeline_command("serve", sys.argv[1:])
