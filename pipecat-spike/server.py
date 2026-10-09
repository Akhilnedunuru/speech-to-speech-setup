"""Phase 3, step 5: WebSocket transport for the Pipecat voice pipeline.

Wires the full rebuilt pipeline behind a WebSocket server:

    WebSocket audio in → VADProcessor (Silero) → RacingSTT
      → BrainProcessor (LangGraph) → TimingProbe → RacingTTS (+ VoiceResolver)
      → WebSocket audio out

VAD note (Pipecat 1.x): the transport takes no vad_analyzer — VAD is a
pipeline processor. VADProcessor emits the VADUserStarted/StoppedSpeaking
frames that SegmentedSTTService (RacingSTT's base) buffers on.

Run:
    PIPECAT_PORT=8767 python server.py        # default port 8767 (prod uses :8765)

Wire format: binary WebSocket frames, 16kHz mono PCM16 both directions.
No serializer: clients send/receive raw PCM16 (telephony serializers like
Twilio/Plivo are for JSON-wrapped protocols, not raw audio).

Env vars: same as spike.py / race.py, plus:
    PIPECAT_PORT        listen port (default 8767)
"""

import asyncio
import logging
import os
import time

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TextFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.task import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from brain import AgenticBrain, BrainProcessor
from groq_adapter import groq_llm_fn
from race import RacingSTT, RacingTTS
from voice import VoiceResolver, load_default_voice_from_env
from raw_pcm_serializer import RawPCMSerializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipecat-server")

PORT = int(os.environ.get("PIPECAT_PORT", "8767"))


def timed_llm_fn(messages):
    """llm_fn wrapper: logs per-think-call latency and outcome type."""
    start = time.perf_counter()
    out = groq_llm_fn(messages)
    logger.info(
        "Brain think: %.2fs -> %s", time.perf_counter() - start, out.get("type")
    )
    return out


class TimingProbe(FrameProcessor):
    """Logs brain-stage latency per turn.

    Placed after BrainProcessor: sees the STT TranscriptionFrame pass through
    (turn start), then the brain's response TextFrame (brain done, TTS about
    to start). STT and TTS race times are already logged by RacingSTT/TTS.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn = 0
        self._t0 = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self._turn += 1
            self._t0 = time.perf_counter()
            logger.info("Turn %d: user said %r", self._turn, frame.text[:100])
        elif isinstance(frame, TextFrame) and self._t0 is not None:
            # Only the brain emits TextFrames here, so this is its final answer.
            logger.info(
                "Turn %d: brain done in %.2fs", self._turn, time.perf_counter() - self._t0
            )
            self._t0 = None
        await self.push_frame(frame, direction)


async def run_bot(websocket: WebSocket):
    """Build and run one pipeline per WebSocket connection."""
    await websocket.accept()
    logger.info("Client connected")

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
    serializer=RawPCMSerializer(sample_rate=16000, num_channels=1),
        ),
    )

    ref_b64, ref_text = load_default_voice_from_env()
    voice_resolver = VoiceResolver(
        default_ref_audio_b64=ref_b64, default_ref_text=ref_text
    )

    stt = RacingSTT()
    brain = AgenticBrain(llm_fn=timed_llm_fn)
    tts = RacingTTS(
        ref_audio_b64=ref_b64,
        ref_text=ref_text,
        voice_resolver=voice_resolver,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            stt,
            BrainProcessor(brain),
            TimingProbe(),
            tts,
            transport.output(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    runner = WorkerRunner()
    try:
        await runner.run(worker)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Pipeline error")
    finally:
        logger.info("Client disconnected")


app = FastAPI()


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "pipecat-voice"})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await run_bot(websocket)


def _check_env():
    missing = [k for k in ("RUNPOD_API_KEY", "GROQ_API_KEY") if not os.environ.get(k)]
    ref_b64, ref_text = load_default_voice_from_env()
    if not ref_b64 or not ref_text:
        missing.append("REF_AUDIO_PATH/REF_TEXT_PATH")
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")


if __name__ == "__main__":
    _check_env()
    logger.info("Pipecat voice server on :%d (WS /ws, health /health)", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
