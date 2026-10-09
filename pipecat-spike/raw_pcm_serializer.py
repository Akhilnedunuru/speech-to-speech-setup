"""Raw PCM serializer for Pipecat WebSocket transport.

Bridges raw 16-bit PCM mono @ 16kHz binary WebSocket frames to Pipecat
audio frames. Pipecat 1.x REQUIRES a serializer — without one, the
transport silently discards all incoming messages.
"""

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer


class RawPCMSerializer(FrameSerializer):
    def __init__(self, sample_rate: int = 16000, num_channels: int = 1):
        super().__init__()
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    async def serialize(self, frame) -> bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data) -> InputAudioRawFrame | None:
        if isinstance(data, (bytes, bytearray)):
            return InputAudioRawFrame(
                audio=bytes(data),
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
            )
        return None
