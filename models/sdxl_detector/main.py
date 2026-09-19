"""
Organika/sdxl-detector wrapper (no reimplementation).
Swin-based image classifier (86.8M params, safetensors), fine-tuned from
umm-maybe/AI-image-detector on Wikimedia-vs-SDXL pairs.
Loaded as-published via transformers AutoImageProcessor +
AutoModelForImageClassification (processor handles resize/normalize).

Label mapping (from the published config.json — verified, NOT assumed):
  id 0 = "artificial"  -> our "fake"
  id 1 = "human"       -> our "real"
fake_probability = softmax(logits)[label2id["artificial"]].
"""
import io, os, sys, time, logging
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Keep HF downloads in the in-repo cache (C: is nearly full).
# Overridable via environment — launch scripts set HF_HOME explicitly.
os.environ.setdefault("HF_HOME", r"D:\DeepGuard\hf_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\DeepGuard\hf_cache\hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\DeepGuard\hf_cache\hub")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sdxl")
app = FastAPI(title="SDXL AI-Image Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = os.getenv("SDXL_MODEL_ID", "Organika/sdxl-detector")

processor = None
model = None
FAKE_LABEL = "artificial"
FAKE_INDEX = 0


@app.on_event("startup")
async def load_model():
    global processor, model, FAKE_INDEX
    logger.info(f"Loading {MODEL_ID} (processor + model as published)")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    label2id = {k.lower(): v for k, v in model.config.label2id.items()}
    assert FAKE_LABEL in label2id, f"expected {FAKE_LABEL!r} in {model.config.label2id}"
    FAKE_INDEX = label2id[FAKE_LABEL]
    logger.info(f"Label map: {model.config.id2label} -> fake='{FAKE_LABEL}' at index {FAKE_INDEX}")
    model = model.to(DEVICE).eval()
    logger.info(f"SDXL detector ready on {DEVICE}")


@app.get("/health")
async def health():
    labels = None
    try:
        labels = model.config.id2label if model is not None else None
    except Exception:
        pass
    return {
        "status": "ok",
        "model": "SDXL",
        "variant": MODEL_ID,
        "device": str(DEVICE),
        "labels": labels,
        "fake_label": FAKE_LABEL,
        "fake_index": FAKE_INDEX,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            fake_prob = torch.softmax(logits, dim=-1)[0, FAKE_INDEX].item()
        return {
            "model": "SDXL",
            "fake_probability": round(fake_prob, 4),
            "verdict": "fake" if fake_prob > 0.5 else "real",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        logger.error(f"SDXL detect error: {e}")
        raise HTTPException(500, detail=str(e))
