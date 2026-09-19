"""
CapCheck — aaronkantrowitz/ai-image-detection deepfake detector (solo eval only).

Binary ViT-Base (85.8M params) fine-tuned from dima806/ai_vs_real_image_detection
on the CIFAKE dataset. Labels: REAL (0) / FAKE (1) — mapping confirmed
behaviorally on CIFAKE anchor images (10/10 correct @ ~0.99 confidence).

/detect returns continuous fake_probability = P(FAKE) from the softmax.
NOT wired into the gateway — solo evaluation only.
"""

import io
import logging
import time

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("capcheck")

app = FastAPI(title="CapCheck Image Deepfake Detector")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

MODEL_ID = "aaronkantrowitz/ai-image-detection"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf = None


@app.on_event("startup")
async def load_model():
    global clf
    from transformers import pipeline

    # NOTE: pipeline runs on CPU unless device is passed; keep default
    # (matches solo-eval usage) but log the torch device for reference.
    clf = pipeline(task="image-classification", model=MODEL_ID, top_k=None)
    logger.info(f"CapCheck ready (torch device: {DEVICE}) "
                f"id2label={clf.model.config.id2label}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "CapCheck/aaronkantrowitz-ai-image-detection",
        "device": str(DEVICE),
        "loaded": clf is not None,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        out = clf(img)
        # Behaviorally confirmed mapping: FAKE score == fake probability.
        # (CIFAKE anchors: FAKE->FAKE @0.997, REAL->REAL @0.997.)
        fake_prob = next(
            float(d["score"]) for d in out if str(d["label"]).upper() == "FAKE"
        )
        return {
            "model": "CapCheck",
            "fake_probability": round(fake_prob, 4),
            "verdict": "fake" if fake_prob > 0.5 else "real",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except StopIteration:
        raise HTTPException(status_code=500, detail="FAKE label missing from output")
    except Exception as e:
        logger.error(f"CapCheck detect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
