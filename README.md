# Deepfake Detector — Setup & Deployment Guide

## Project structure

```
deepfake-detector/
├── gateway/
│   └── main.py              ← FastAPI gateway :8000
├── models/
│   ├── npr/main.py          ← NPR image model :5001
│   ├── ufd/main.py          ← UniversalFakeDetect image :5004
│   ├── rawnet/main.py       ← RawNet2 audio :5002
│   └── crossvit/main.py     ← CrossEfficientViT video :7001
├── frontend/                ← Put your built React SPA here
├── nginx/nginx.conf         ← Reverse proxy config
├── weights/                 ← Put your .pth model weights here
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.gateway
├── Dockerfile.model
├── start.sh                 ← Local dev launcher
└── stop.sh
```

---

## Option A — Local development (no Docker)

### 1. Prerequisites

- Python 3.10 or 3.11
- pip
- ffmpeg (for video processing)

```bash
# macOS
brew install ffmpeg python@3.11

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg python3.11 python3-pip
```

### 2. Create a virtual environment

```bash
cd deepfake-detector
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> GPU support: replace `torch==2.3.0` in requirements.txt with the CUDA build:
> `pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121`

### 4. Add model weights

Run the downloader (skips files already present, warms the Hugging Face
cache for the HF-hosted models):

```
venv\Scripts\python download_weights.py
```

Weight sources per model:

| Model | File | Source | How obtained |
|---|---|---|---|
| NPR | `weights/npr.pth` (282 MB) | CNNDetection GitHub release | scripted |
| UFD | `weights/ufd.pth` (4 KB) | UniversalFakeDetect GitHub release (+ `openai-clip` pip package) | scripted |
| IAPL backbone | `weights/ViT-L-14.pt` (933 MB) | OpenAI public CDN | scripted |
| Nonescape | `weights/nonescape-v0.safetensors` (2.4 GB) | Nonescape CDN (DigitalOcean) | scripted |
| SDXL-detector | HF cache (auto) | `Organika/sdxl-detector` on Hugging Face | scripted (pre-warmed by the script) |
| umm-maybe | HF cache (auto) | `umm-maybe/AI-image-detector` on Hugging Face | scripted (pre-warmed by the script) |
| CapCheck | HF cache (auto) | `aaronkantrowitz/ai-image-detection` on Hugging Face | scripted (pre-warmed by the script — no separate manual step) |
| CrossViT | `weights/crossvit.pth` (1.1 GB) | Coccomini et al. GitHub release | scripted entry (service unevaluated) |
| **IAPL checkpoint** | **`weights/iapl_sd14.pth` (~1.7 GB)** | **https://modelscope.cn/models/yihengli/IAPL_pretrain** | **MANUAL — no verified scriptable source (ModelScope page is JS-gated). Download the SDv1.4 checkpoint and save it exactly here or IAPL will refuse to start.** |
| RawNet2 | `weights/rawnet2.pth` | none working (listed Zenodo URL serves HTML) | MANUAL — service stays down until a genuine file is placed |

Notes:
- HF-hosted models honour `HF_HOME` / `HF_HUB_CACHE`. The launch scripts
  (`install_and_run.bat`, `start_all.bat`) set these to
  `D:\DeepGuard\hf_cache` automatically — all Hugging Face downloads land
  there, never on C: (which is nearly full on this machine). The resolved
  location is `D:\DeepGuard\hf_cache\hub`; `download_weights.py` prints it
  on every run. To manually clear or inspect the cache, that is the path —
  not `C:\Users\<you>\.cache\huggingface`. The directory is gitignored.
- Then update each `models/<name>/main.py` — find the `TODO` block in `load_model()` and uncomment / adapt the real loading code.

### 5. Start all services

```bat
install_and_run.bat
```
(quick relaunch without reinstalls: `start_all.bat`)

Services:
| Service | URL |
|---|---|
| Gateway API | http://localhost:8000 |
| Interactive docs | http://localhost:8000/docs |
| NPR model | http://localhost:5001 |
| UFD model | http://localhost:5004 |
| IAPL model | http://localhost:5005 |
| SDXL-detector | http://localhost:5009 |
| umm-maybe | http://localhost:5010 |
| CapCheck model | http://localhost:5011 |
| Nonescape model | http://localhost:5013 |
| CrossViT | http://localhost:7001 |
| RawNet2 | http://localhost:5002 (down — no usable weights, see above) |

### 6. Test the API

```bash
# Image detection
curl -X POST http://localhost:8000/api/detect \
  -F "file=@test_image.jpg" \
  -F "media_type=image" \
  -F "models=npr,ufd" \
  -F "strategy=stacking"

# Audio detection
curl -X POST http://localhost:8000/api/detect \
  -F "file=@test_audio.wav" \
  -F "media_type=audio" \
  -F "strategy=voting"

# Video detection
curl -X POST http://localhost:8000/api/detect \
  -F "file=@test_video.mp4" \
  -F "media_type=video" \
  -F "strategy=average"
```

Expected response:
```json
{
  "filename": "test_image.jpg",
  "media_type": "image",
  "strategy": "stacking",
  "verdict": "fake",
  "fake_probability": 0.8312,
  "confidence": 0.8312,
  "model_results": [
    { "model": "NPR", "verdict": "fake", "fake_probability": 0.79, "latency_ms": 42 },
    { "model": "UniversalFakeDetect", "verdict": "fake", "fake_probability": 0.87, "latency_ms": 61 }
  ]
}
```

### 7. Stop all services

```bash
./stop.sh
```

---

## Option B — Docker Compose (recommended for production)

### 1. Prerequisites

- Docker 24+
- Docker Compose v2

```bash
# Ubuntu
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER   # re-login after this
```

### 2. Build and start

```bash
cd deepfake-detector

# First time (builds all images)
docker compose up --build

# Background mode
docker compose up -d --build
```

### 3. Check status

```bash
docker compose ps
docker compose logs gateway     # tail gateway logs
docker compose logs -f crossvit # follow video model logs
```

### 4. Stop

```bash
docker compose down
```

---

## Connecting the frontend

### Update the frontend API call

In your React app, replace the mock `runDetection()` with:

```javascript
async function runDetection(file, mediaType, models, strategy) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("media_type", mediaType);           // "image" | "audio" | "video"
  formData.append("models", models.join(","));         // "npr,ufd"
  formData.append("strategy", strategy);              // "stacking" | "voting" | "average"

  const res = await fetch("/api/detect", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

When running locally without Docker, use `http://localhost:8000/api/detect` directly.
With Docker / Nginx, use `/api/detect` (the proxy handles it).

---

## Adding a new model

1. Create `models/yourmodel/main.py` (copy any existing model as template)
2. Implement `load_model()` and `run_model()` — return `fake_probability` in [0, 1]
3. Add the service to `gateway/main.py` under `MODEL_SERVICES`
4. Add a new service block in `docker-compose.yml`

---

## Environment variables (gateway)

| Variable | Default | Description |
|---|---|---|
| `NPR_URL` | `http://localhost:5001/detect` | NPR model endpoint |
| `UFD_URL` | `http://localhost:5004/detect` | UFD model endpoint |
| `RAWNET_URL` | `http://localhost:5002/detect` | RawNet2 endpoint |
| `CROSSVIT_URL` | `http://localhost:7001/detect` | CrossViT endpoint |

Set these in `.env` or in `docker-compose.yml` under `environment:`.

---

## Common issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: librosa` | `pip install librosa soundfile` |
| `cv2` not found | `pip install opencv-python-headless` |
| CUDA out of memory | Reduce batch size or use CPU (`map_location="cpu"`) |
| Port already in use | `lsof -i :8000` then `kill <pid>` |
| Video model slow | Reduce `FRAMES_TO_SAMPLE` in `crossvit/main.py` |
