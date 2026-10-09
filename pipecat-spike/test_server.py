"""Mocked tests for the Phase 3 transport step.

- groq_adapter tests run anywhere (no pipecat dependency).
- Pipeline assembly test requires pipecat; skipped gracefully without it
  (runs on the VM, which has pipecat-ai 1.12.0 installed).
"""

import asyncio
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

import groq_adapter
from groq_adapter import _extract_tool_call, groq_llm_fn, to_chat_messages

try:
    import pipecat  # noqa: F401

    PIPECAT_AVAILABLE = True
except ImportError:
    PIPECAT_AVAILABLE = False


def _fake_completion(text):
    """Build a fake openai chat.completions.create return value."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestToolCallExtraction(unittest.TestCase):
    def test_plain_json_tool_call(self):
        out = _extract_tool_call('{"tool": "get_weather", "args": {"city": "Dallas"}}')
        self.assertEqual(out, ("get_weather", {"city": "Dallas"}))

    def test_fenced_json_tool_call(self):
        out = _extract_tool_call(
            '```json\n{"tool": "get_weather", "args": {"city": "Austin"}}\n```'
        )
        self.assertEqual(out, ("get_weather", {"city": "Austin"}))

    def test_plain_text_is_not_a_tool(self):
        self.assertIsNone(_extract_tool_call("The weather is nice today."))
        self.assertIsNone(_extract_tool_call('{"temp": 72}'))  # no "tool" key
        self.assertIsNone(_extract_tool_call("{not json"))

    def test_missing_args_defaults_to_empty(self):
        out = _extract_tool_call('{"tool": "get_weather"}')
        self.assertEqual(out, ("get_weather", {}))


class TestMessageConversion(unittest.TestCase):
    def test_tool_role_rewritten(self):
        msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "tool", "name": "get_weather", "content": "72F sunny"},
        ]
        out = to_chat_messages(msgs)
        self.assertEqual(out[0]["role"], "user")
        self.assertEqual(out[1]["role"], "user")
        self.assertIn("get_weather", out[1]["content"])
        self.assertIn("72F sunny", out[1]["content"])
        # input not mutated
        self.assertEqual(msgs[1]["role"], "tool")


class TestGroqLlmFn(unittest.TestCase):
    def _patch_client(self, reply_text):
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_completion(reply_text)
        return patch.object(groq_adapter, "_get_client", return_value=client)

    def test_response_passthrough(self):
        with self._patch_client("Hello there!"):
            out = groq_llm_fn([{"role": "user", "content": "hi"}])
        self.assertEqual(out, {"type": "response", "text": "Hello there!"})

    def test_tool_call_translation(self):
        with self._patch_client('{"tool": "get_weather", "args": {"city": "Dallas"}}'):
            out = groq_llm_fn([{"role": "user", "content": "weather in Dallas?"}])
        self.assertEqual(
            out, {"type": "tool", "name": "get_weather", "args": {"city": "Dallas"}}
        )

    def test_system_prompt_prepended(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_completion("ok")
        with patch.object(groq_adapter, "_get_client", return_value=client):
            groq_llm_fn([{"role": "user", "content": "hi"}])
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("get_weather", sent[0]["content"])
        self.assertEqual(sent[1], {"role": "user", "content": "hi"})


@unittest.skipUnless(PIPECAT_AVAILABLE, "pipecat not installed")
class TestPipelineAssembly(unittest.TestCase):
    """Verify the full pipeline assembles: transport -> STT -> brain -> TTS.

    Runs on the VM (pipecat installed). Construction only — no network.
    """

    def test_assemble(self):
        import server

        # Stub the websocket; transport only holds it until run.
        transport_holder = {}

        class FakeWebSocket:
            pass

        with patch.object(
            server, "FastAPIWebsocketTransport", autospec=True
        ) as mock_transport_cls, patch.object(
            server, "RacingSTT", autospec=True
        ), patch.object(
            server, "RacingTTS", autospec=True
        ), patch.object(
            server, "VoiceResolver", autospec=True
        ), patch.object(
            server, "load_default_voice_from_env", return_value=("b64", "text")
        ), patch.object(
            server, "AgenticBrain", autospec=True
        ), patch.object(
            server, "BrainProcessor", autospec=True
        ), patch.object(
            server.TimingProbe, "process_frame", autospec=True
        ):
            transport = mock_transport_cls.return_value
            transport_holder["t"] = transport

            async def fake_run_bot(ws):
                await server.run_bot(ws)

            # run_bot accepts the websocket then builds the pipeline; the
            # WorkerRunner is what would block, so stub it.
            with patch.object(server, "WorkerRunner", autospec=True) as mock_runner_cls, patch.object(
                server, "VADProcessor", autospec=True
            ), patch.object(server, "SileroVADAnalyzer", autospec=True):
                async def fake_accept():
                    return None

                ws = FakeWebSocket()
                ws.accept = fake_accept
                asyncio.run(fake_run_bot(ws))

                # Transport constructed for raw PCM16, no serializer, no VAD
                # param (VAD is a pipeline processor in Pipecat 1.x).
                _, kwargs = mock_transport_cls.call_args
                params = kwargs["params"]
                self.assertTrue(params.audio_in_enabled)
                self.assertTrue(params.audio_out_enabled)
                self.assertFalse(params.add_wav_header)
                self.assertIsNone(params.serializer)
                self.assertFalse(hasattr(params, "vad_analyzer"))
                # Pipeline was handed to a worker and run
                self.assertTrue(mock_runner_cls.return_value.run.called)

    def test_disconnect_does_not_crash(self):
        """A failing runner still logs and exits cleanly."""
        import server

        async def boom_run_bot(ws):
            with patch.object(server, "WorkerRunner", autospec=True) as mock_runner_cls:
                mock_runner_cls.return_value.run.side_effect = RuntimeError("ws gone")
                with patch.object(server, "FastAPIWebsocketTransport", autospec=True), patch.object(
                    server, "RacingSTT", autospec=True
                ), patch.object(server, "RacingTTS", autospec=True), patch.object(
                    server, "VoiceResolver", autospec=True
                ), patch.object(
                    server, "load_default_voice_from_env", return_value=("b64", "text")
                ), patch.object(server, "AgenticBrain", autospec=True), patch.object(
                    server, "BrainProcessor", autospec=True
                ), patch.object(
                    server, "VADProcessor", autospec=True
                ), patch.object(
                    server, "SileroVADAnalyzer", autospec=True
                ):

                    async def fake_accept():
                        return None

                    ws = MagicMock()
                    ws.accept = fake_accept
                    await server.run_bot(ws)  # must not raise

        asyncio.run(boom_run_bot(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
