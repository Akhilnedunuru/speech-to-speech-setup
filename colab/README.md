# Colab path (alternative)

Same voice pipeline, running on a Colab T4 GPU instead of the hybrid
(Oracle + serverless GPU) setup described in the top-level README.

**When to use this:** zero-commitment dev and experiments. Nothing to deploy —
open the notebook, run the cells, talk through the Mac client. Good for trying
model swaps, debugging a stage, or demos.

**When not to:** anything that needs to stay up. Colab reclaims the VM
(typically within hours, always within ~24h), and every fresh session
re-downloads models and re-establishes the ngrok tunnels.

The notebook (`speech-to-speech-colab.ipynb`):
- GPU stages on the T4: Parakeet TDT STT + Qwen3-TTS (the same models the
  hybrid's RunPod legs run)
- LLM is remote either way (`openai/gpt-oss-20b:groq` via the HF router)
- Two ngrok tunnels: stable speech endpoint (`wss://…/v1/realtime`) + a
  second tunnel for the tool API the Mac client forwards to
- Secrets via Colab Secrets: `HF_TOKEN`, `NGROK_TOKEN`, `NGROK_DOMAIN`
