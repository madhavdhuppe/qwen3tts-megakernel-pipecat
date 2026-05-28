# Vast.ai RTX 5090 Runbook

Use fake mode locally. Use this runbook only after the Vast.ai instance is running.

## 1. Pick Instance

Choose:

- GPU: RTX 5090
- CUDA: 12.8 or newer
- Python: 3.10 or 3.11
- Disk: at least 40 GB
- RAM: at least 32 GB

Avoid spending time on real mode until `nvidia-smi` confirms RTX 5090.

## 2. Copy Repo

From your local machine:

```bash
rsync -av --exclude venv --exclude .git \
  /Users/anjalidhuppe/Desktop/qwen3tts-megakernel-pipecat/ \
  root@YOUR_VAST_HOST:/workspace/qwen3tts-megakernel-pipecat/
```

Or clone/pull your Git repo if you push this project first.

## 3. Create Env On Vast

```bash
cd /workspace/qwen3tts-megakernel-pipecat
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If the image does not already include a CUDA-enabled PyTorch for Blackwell, install the matching CUDA 12.8 PyTorch wheel before `requirements.txt`.

## 4. Verify GPU First

```bash
source venv/bin/activate
python scripts/verify_5090_env.py
```

You want to see:

- `gpu_name` contains `RTX 5090`
- `compute_capability=12.0` or newer
- `cuda_available=True`
- `bf16_matmul_ok=(1024, 1024)`

## 5. Tiny Real-Mode Smoke

```bash
export MEGAKERNEL_TTS_MODE=real
python demo/demo.py \
  --mode real \
  --text "Hello hi AI" \
  --output output/hello_hi_ai_real.wav
```

The first run can take about a minute because the CUDA extension JIT-compiles and model weights may download.

## 6. Benchmark

```bash
python benchmark/benchmark.py --mode real --runs 3 --text "Hello hi AI"
```

Record:

- TTFC
- RTF
- total chunks
- whether the WAV file sounds like speech

## 7. Server Smoke

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

In another shell:

```bash
curl -N -X POST http://127.0.0.1:8000/tts/wav \
  -H "content-type: application/json" \
  -d '{"text":"Hello hi AI","mode":"real"}' \
  --output output/server_real.wav
```

Stop if any earlier step fails. Fix environment before debugging Pipecat or server code.
