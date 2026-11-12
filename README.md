# InkSight AI

InkSight AI converts handwritten English images into editable text. It uses the locally downloaded `microsoft/trocr-base-handwritten` model through Transformers and PyTorch when those packages can load. Images are processed in memory; SQLite stores text and metadata only.

## Features and architecture

- React, TypeScript, Vite, Tailwind, Lucide and Axios interface with dashboard, recognition, history, settings, and academic project overview.
- FastAPI health/model status, recognition, SQLite history, and live usage statistics.
- JPG/PNG validation, 15 MB upload limit, 30 MP image limit, EXIF orientation correction, resizing, grayscale denoising, adaptive thresholding, horizontal line joining, and ordered line segmentation.
- Auto line detection and explicit single-region fallback. Paragraph separation remains approximate.
- Local inference is the default; `INFERENCE_PROVIDER=huggingface` enables a token-configured online option that is labeled in the UI.

The React UI calls `backend/app/main.py`; image processing is in `backend/app/services/images.py`; `recognition.py` isolates provider logic. SQLite history defaults to `backend/data/history.db`. The model stays outside the project in `../models/trocr-base-handwritten` and is loaded once at API startup.

## Project structure

```text
HandwritingAI/
├── backend/app/                 FastAPI routes and recognition/image services
├── backend/requirements*.txt    Standard and policy-compatible backend dependencies
├── frontend/src/                React, TypeScript, styles, and application pages
├── scripts/                     Environment and model diagnostics
├── tests/                       Image processing and API tests
├── .env.example                 Safe configuration template
├── start_backend.ps1
└── start_frontend.ps1
```

Virtual environments, `node_modules`, build output, local history, `.env`, model weights, and caches are excluded from Git.

## Setup

Use the existing Python 3.11 environment and Node.js 20+. The model directory is detected by default. To use the policy-compatible API environment without replacing `.venv`, install in PowerShell:

```powershell
cd C:\Users\ABHAY\OneDrive\Desktop\Clg_Mini_Project\HandwritingAI
Copy-Item .env.example .env
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
py -3.11 -m venv .backend-pure-venv
$env:SKIP_CYTHON = '1'
.\.backend-pure-venv\Scripts\python.exe -m pip install --no-binary=pydantic -r backend\requirements-policy-compatible.txt
cd frontend
npm install
```

## PyTorch / Windows Application Control finding

The environment reports Python 3.11.9, `torch 2.15.0`, `torchvision 0.29.2`, and Transformers 5.22.2. The [PyTorch/TorchVision release compatibility notes](https://github.com/pytorch/vision/releases) say TorchVision 0.29 is ABI compatible with PyTorch 2.14 and later, so the observed pair is compatible; Python 3.11 is supported by the installed wheels. Transformers 5 uses its current multimodal auto-model class, while the loader keeps a Transformers 4 path as well. Importing torch reproduces `OSError: [WinError 4551] An Application Control policy has blocked this file` at `torch\lib\c10.dll`; this is an explicit Windows policy block, not evidence of a package-version mismatch. Code Integrity events 3033 and 3077 name policy ID `0283ac0f-fff1-49ae-ada1-8a933130cad6` and state the files did not meet enterprise signing requirements. The files confirmed as unsigned are `torch\lib\c10.dll`, `pydantic_core\_pydantic_core.cp311-win_amd64.pyd`, and `frontend\node_modules\@rollup\rollup-win32-x64-msvc\rollup.win32-x64-msvc.node`. The existing `.venv` cannot import Pydantic 2 because Pydantic Core is blocked. For API development, the project has an isolated `.backend-pure-venv` with FastAPI 0.125 and Pydantic 1.10 built in pure-Python mode; FastAPI 0.125 supports Pydantic v1. The startup script selects it while leaving `.venv` intact. Vite uses Rollup's official `@rollup/wasm-node`, which has no native code, and the production build succeeds. These alternatives let the backend and frontend run, but do not bypass the PyTorch block. An administrator must review the policy and approve the trusted PyTorch package or publisher for local TrOCR inference. Do not disable security protections. Backend status reports model unavailable rather than fabricating output. `scripts/diagnose.py` contains further checks.

## Run

Open separate PowerShell windows at the project root:

```powershell
.\start_backend.ps1
```

```powershell
.\start_frontend.ps1
```

Open http://localhost:5173. API health: http://127.0.0.1:8000/api/health; API docs: http://127.0.0.1:8000/docs.

## Configuration

Copy `.env.example` to `.env`. `MODEL_PATH` defaults to the existing model. `INFERENCE_PROVIDER` accepts `local` or `huggingface`. Online mode requires `HF_TOKEN` and may use `HF_API_MODEL`; never commit `.env`. Hosted inference depends on endpoint availability and credentials and is not local inference. Upload defaults: 15 MB and 30 megapixels.

## Model setup

The model is not included in this repository (the weights are over 1 GB). Download `microsoft/trocr-base-handwritten` once using the Hugging Face CLI or `huggingface_hub.snapshot_download`, placing it at `../models/trocr-base-handwritten` relative to this project, or set `MODEL_PATH` in `.env` to your existing model directory. Required files include `config.json`, `preprocessor_config.json`, tokenizer files, and either `model.safetensors` or `pytorch_model.bin`. The app uses only local model files in local mode and does not download the model automatically.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | API, Python, provider, device, and model availability |
| GET | `/api/model-status` | Model readiness and diagnostic message |
| GET | `/api/stats` | Usage statistics from local history |
| POST | `/api/recognize` | Upload JPG/PNG (`file`) and select `segmentation_mode` (`auto` or `single`) |
| GET | `/api/history` | List saved recognition records |
| DELETE | `/api/history/{id}` | Delete one history record |
| DELETE | `/api/history` | Clear recognition history |

## Tests and diagnostics

```powershell
.\.venv\Scripts\python.exe scripts\diagnose.py
.\.venv\Scripts\python.exe -m pytest tests\test_images.py -q
```

Tests cover preprocessing, segmentation, invalid uploads, API status, and history. TrOCR recognition requires a genuinely loadable model; tests never substitute fabricated recognition output.

## Known limitations

TrOCR is primarily a line recognition model. Adaptive/Otsu thresholding plus horizontal morphology can struggle with skew, overlapping rows, unusual handwriting, or poor lighting. A single-region fallback may be less accurate on a full page. Paragraph layout is approximate. On this computer local loading is blocked by the reported Windows policy. Hosted inference requires a working route and credentials.
