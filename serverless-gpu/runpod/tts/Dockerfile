# RunPod serverless image: Qwen3-TTS 1.7B CustomVoice (CUDA, ggml backend)
# Build for linux/amd64 (RunPod GPUs are x86_64):
#   docker buildx build --platform linux/amd64 -t <dockerhub-user>/s2s-qwen3-tts:latest --push runpod/tts
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch first (CUDA 12.4 wheels), then faster-qwen3-tts with its ggml extra
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir "faster-qwen3-tts[ggml]" runpod soundfile numpy scipy

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
