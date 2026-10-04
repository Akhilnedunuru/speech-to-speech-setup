# RunPod serverless image: Parakeet TDT 0.6B STT (CUDA)
# Build for linux/amd64 (RunPod GPUs are x86_64):
#   docker buildx build --platform linux/amd64 -t <dockerhub-user>/s2s-parakeet-stt:latest --push runpod/stt
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch first (CUDA 12.4 wheels), then the rest
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir runpod nano-parakeet soundfile numpy scipy

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
