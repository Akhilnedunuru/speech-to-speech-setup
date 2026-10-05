# RunPod serverless image: Qwen3-TTS 1.7B CustomVoice (CUDA, ggml backend)
# Build for linux/amd64 (RunPod GPUs are x86_64):
#   docker buildx build --platform linux/amd64 -t <dockerhub-user>/s2s-qwen3-tts:latest --push runpod/tts
# NOTE: base image must be Ubuntu 24.04 (glibc 2.39+): qwentts-cpp-python
# only ships manylinux_2_39 wheels, which pip refuses on Ubuntu 22.04's
# glibc 2.35 ("No matching distribution found"). CUDA 12.8 also matches the
# wheel's cu128 build; torch cu124 runs fine on a 12.8 runtime.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

# PyTorch first (CUDA 12.4 wheels), then numpy on its own: the transitive
# `sox` package runs `import sox` (which imports numpy) inside its setup.py
# at egg_info time, so numpy must already be installed or the build dies.
RUN pip3 install --break-system-packages --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --break-system-packages --no-cache-dir numpy
# NOTE: version pinned — the handler targets the 0.5.x API (backend="ggml").
# An unpinned install once silently pulled 0.2.6 (torch-only), whose
# from_pretrained() has no `backend` kwarg and crashes the worker on startup.
RUN pip3 install --break-system-packages --no-cache-dir "faster-qwen3-tts[ggml]==0.5.3" runpod soundfile scipy

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
