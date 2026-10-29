# BHUVISION

BHUVISION is a local satellite-image analysis app with a React/Vite frontend and a FastAPI backend. Inference uses model files stored in this repository. The application runs with Hugging Face offline mode enabled and does not call AWS, hosted model APIs, or other inference services.

## Included model weights

About 2.2 GB of model files are stored with Git LFS under `models/`:

| Model | Use | Approx. size | License |
| --- | --- | ---: | --- |
| Salesforce BLIP VQA Base | Image questions and descriptions | 1.54 GB | BSD-3-Clause |
| IDEA Research Grounding DINO Tiny | Open vocabulary object detection | 689 MB | Apache-2.0 |

See [models/README.md](models/README.md) for model sources and license notices.

## Installation on Windows

Install Git, Git LFS, Python 3.10 or 3.11, and Node.js 20.19+ or 22.12+. Git LFS is required to download the model files.

```powershell
git lfs install
git clone https://github.com/shantanudwivedi1/BHUVISION.git
cd BHUVISION
git lfs pull

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Copy-Item .env.example .env
npm install
```

PyTorch installation may differ for CPU and CUDA machines. If pip cannot resolve a matching torch build, follow the [official PyTorch install selector](https://pytorch.org/get-started/locally/) and install the matching `torch` and `torchvision` pair before installing the remaining requirements.

## Run the app

Open two PowerShell terminals in the repository. Activate the virtual environment in the backend terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in the second terminal:

```powershell
npm run dev -- --host 127.0.0.1
```

Open <http://127.0.0.1:5173/>. The backend health endpoint is <http://127.0.0.1:8000/health>.

## Linux and macOS setup

Install Git LFS, Python 3.10 or 3.11, and Node.js 20.19+ or 22.12+, then run:

```bash
git lfs install
git clone https://github.com/shantanudwivedi1/BHUVISION.git
cd BHUVISION
git lfs pull

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
npm install
```

Run the backend and frontend in separate terminals using the same commands shown above, replacing the PowerShell activation command with `source .venv/bin/activate`.

## Offline use

The model weights are included in Git LFS, so the initial clone and dependency installation need network access. After Git LFS has downloaded the files and Python and Node dependencies are installed, the app can run without internet access. `backend.main` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; model loaders use local files only. Missing weights are reported as unavailable and are not downloaded in the background.

When transferring a working installation to an offline computer, copy the entire repository including the materialized LFS files, the Python environment or wheel packages, and `node_modules` if they are needed there. Do not copy `.env` if it contains local secrets.

## Troubleshooting

- If model files appear as small text pointer files, install Git LFS and run `git lfs pull` while online.
- If PowerShell blocks virtual environment activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
- If PyTorch cannot use CUDA, install the torch build matching the installed CUDA runtime or set the model device options in `.env` to `cpu`.
- The first inference loads the local model into memory and can take a little time. Later requests reuse the loaded model.
