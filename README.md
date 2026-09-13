# BHUVISION

BHUVISION is a local FastAPI + React prototype. The default path is deterministic planning plus lightweight image metadata. CROMA is an optional GPU capability and is disabled by default; no checkpoint is downloaded on the development machine.

## Run the local demo

From the repository root, use two terminals:

```bash
# Terminal 1: backend
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
# Terminal 2: frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/`. The frontend proxies `/health` and `/analyze` to the backend. Verify the backend independently with:

```bash
curl http://127.0.0.1:8000/health
```

## CROMA capability

CROMA is integrated as a Transformers-based optical + SAR feature extractor. It expects co-registered Sentinel-1 SAR input with shape `(2, H, W)` and Sentinel-2 optical input with shape `(12, H, W)`. A 13-band Sentinel-2 tensor is accepted after removing B10. The adapter returns a learned joint representation shape only; it does not produce semantic labels or confidence scores.

The implementation follows the Transformers CROMA model card at [`BiliSakura/CROMA-transformers`](https://huggingface.co/BiliSakura/CROMA-transformers), using `AutoImageProcessor` and `AutoModel` with `trust_remote_code=True`.

## Run CROMA on a GPU host

Run these commands on a CUDA-enabled Linux host or cloud GPU runtime, not the MacBook Air development environment:

```bash
python -m venv .venv-croma
source .venv-croma/bin/activate

# Install the torch build matching the host CUDA version.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers einops numpy opencv-python fastapi uvicorn

export BHUVISION_CROMA_ENABLED=true
export BHUVISION_CROMA_DEVICE=cuda
export BHUVISION_CROMA_MODEL_ID=BiliSakura/CROMA-transformers
export BHUVISION_CROMA_ALLOW_DOWNLOAD=true
export BHUVISION_CROMA_ALLOW_CPU=false

uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The first enabled run downloads the configured Hugging Face checkpoint to that GPU host. For a pre-staged checkpoint, set `BHUVISION_CROMA_MODEL_PATH=/absolute/path/to/croma-base-patch8-120` and leave `BHUVISION_CROMA_ALLOW_DOWNLOAD=false`.

The current `/analyze` upload path does not infer Sentinel-1/Sentinel-2 multiband tensors from ordinary PNG/JPEG files, so it does not invoke CROMA automatically. Use `CROMAAdapter.run(sar_images, optical_images)` with modality-labeled arrays when wiring the specialist execution path.
