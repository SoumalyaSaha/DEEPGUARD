"""
FastAPI Gateway v8 — :8000
7-model majority-vote deepfake detection + persistent audit log + thumbnails.
"""

import asyncio
import hashlib
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from typing import Optional
import io
import json
import logging
import os
import random
import string
import time
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

app = FastAPI(title="Deepfake Detection Gateway", version="8.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "logs"))
THUMBS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "thumbnails"))
DETECTIONS_LOG = os.path.join(LOGS_DIR, "detections_log.jsonl")
FEEDBACK_LOG = os.path.join(LOGS_DIR, "feedback_log.jsonl")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(THUMBS_DIR, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMBS_DIR), name="thumbnails")

MODEL_SERVICES = {
    "image": [
        {"id": "npr", "url": os.getenv("NPR_URL", "http://localhost:5001/detect"), "name": "NPR"},
        {"id": "ufd", "url": os.getenv("UFD_URL", "http://localhost:5004/detect"), "name": "UFD"},
        {"id": "iapl", "url": os.getenv("IAPL_URL", "http://localhost:5005/detect"), "name": "IAPL"},
        {"id": "sdxl", "url": os.getenv("SDXL_URL", "http://localhost:5009/detect"), "name": "SDXL"},
        {"id": "umm", "url": os.getenv("UMM_URL", "http://localhost:5010/detect"), "name": "UMM"},
        {"id": "capcheck", "url": os.getenv("CAP_URL", "http://localhost:5011/detect"), "name": "CapCheck"},
        {"id": "nonescape", "url": os.getenv("NONE_URL", "http://localhost:5013/detect"), "name": "Nonescape"},
    ],
    "audio": [
        {"id": "rawnet", "url": os.getenv("RAWNET_URL", "http://localhost:5002/detect"), "name": "RawNet2"},
    ],
    "video": [
        {"id": "crossvit", "url": os.getenv("CROSSVIT_URL", "http://localhost:7001/detect"), "name": "CrossEfficientViT"},
    ],
}

MODEL_DISPLAY = {
    "npr": ("NPR", "Noise Pattern Recognition"),
    "ufd": ("UFD", "Universal Fake Detection"),
    "iapl": ("IAPL", "Inception Anomaly Detection"),
    "sdxl": ("SDXL_DETECTOR", "Diffusion Artifact Detection"),
    "umm": ("DIFFUSION_RECON", "Error Variance Inversion"),
    "capcheck": ("CAPCHECK", "Caption Artifact Detection"),
    "nonescape": ("NONESCAPE", "Spectral Residual Analysis"),
}

MODEL_WEIGHTS = {"npr": 1.00, "ufd": 1.00, "iapl": 0.844}
TIMEOUT = httpx.Timeout(60.0)
_ALPHANUM = string.ascii_uppercase + string.digits
VOTE_THRESHOLD = 0.5


def _gen_verification_id():
    return "VA-" + "".join(random.choices(_ALPHANUM, k=4)) + "-" + "".join(random.choices(_ALPHANUM, k=4))


# -- Audit log helpers --

def _write_detection(record):
    with open(DETECTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _load_all_detections():
    records = []
    if not os.path.exists(DETECTIONS_LOG):
        return records
    with open(DETECTIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    records.sort(key=lambda r: r.get("timestamp_utc", ""), reverse=True)
    return records


def _load_detection_by_id(verification_id):
    if not os.path.exists(DETECTIONS_LOG):
        return None
    with open(DETECTIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("verification_id") == verification_id:
                    return rec
            except json.JSONDecodeError:
                continue
    return None


def _update_detection_flag(feedback_token, field, value):
    if not os.path.exists(DETECTIONS_LOG):
        return False
    lines = []
    updated = False
    with open(DETECTIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line)
            if updated:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                if rec.get("feedback_token") == feedback_token:
                    rec[field] = value
                    lines[-1] = json.dumps(rec, default=str) + "\n"
                    updated = True
            except json.JSONDecodeError:
                continue
    if updated:
        with open(DETECTIONS_LOG, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return updated


def _save_thumbnail(file_bytes, verification_id):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.thumbnail((120, 120), Image.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        thumb_name = verification_id.lower() + ".jpg"
        thumb_path = os.path.join(THUMBS_DIR, thumb_name)
        img.save(thumb_path, "JPEG", quality=85)
        return "/thumbnails/" + thumb_name
    except Exception as e:
        logger.warning("Thumbnail save failed: %s", e)
        return None


def _file_ext(filename):
    ext = os.path.splitext(filename or "")[1].upper().lstrip(".")
    return ext if ext in ("PNG", "JPEG", "JPG", "WEBP", "TIFF", "BMP") else "UNKNOWN"


def _enrich_breakdown(raw_results):
    enriched = []
    for r in raw_results:
        model_id = r.get("model_id", "")
        display_key, label = MODEL_DISPLAY.get(model_id, (r["model"], r["model"]))
        entry = {
            "model": display_key,
            "internal_name": r["model"],
            "label": label,
            "score": round(r["fake_prob"], 4) if r["verdict"] != "error" else None,
            "vote": "fake" if r.get("fake_prob", 0) > 0.5 and r["verdict"] != "error" else ("real" if r["verdict"] != "error" else "error"),
            "display_type": "variance" if model_id == "nonescape" else "probability",
        }
        if model_id == "nonescape" and r["verdict"] != "error":
            entry["threshold"] = 0.5
        enriched.append(entry)
    return enriched


# -- In-memory feedback cache (1h TTL) --

FEEDBACK_TTL_S = 3600
_detections_cache = {}


def _prune_detections(now=None):
    now = time.time() if now is None else now
    expired = [t for t, v in _detections_cache.items() if now - v["timestamp"] > FEEDBACK_TTL_S]
    for t in expired:
        del _detections_cache[t]


class FeedbackIn(BaseModel):
    feedback_token: str
    user_verdict: str


# -- Helpers --

async def call_model(client, service, file_bytes, filename):
    try:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        resp = await client.post(service["url"], files=files, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return {
            "model": service["name"],
            "model_id": service["id"],
            "fake_prob": float(data.get("fake_probability", data.get("fake_prob", 0.5))),
            "verdict": data.get("verdict", "unknown"),
            "latency_ms": data.get("latency_ms", None),
            "error": None,
        }
    except Exception as e:
        logger.warning("Model %s failed: %s", service["name"], e)
        return {
            "model": service["name"],
            "model_id": service["id"],
            "fake_prob": 0.5,
            "verdict": "error",
            "latency_ms": None,
            "error": str(e),
        }


def majority_vote(valid_results):
    responded = len(valid_results)
    if responded == 0:
        raise ValueError("No predictions to combine")
    votes_fake = sum(1 for r in valid_results if r["fake_prob"] > VOTE_THRESHOLD)
    votes_real = responded - votes_fake
    needed = responded // 2 + 1
    verdict = "fake" if votes_fake >= needed else "real"
    agree = max(votes_fake, votes_real)
    breakdown = [
        {"model": r["model"], "score": round(r["fake_prob"], 4),
         "vote": "fake" if r["fake_prob"] > VOTE_THRESHOLD else "real"}
        for r in valid_results
    ]
    return {
        "verdict": verdict,
        "confidence_label": "%d of %d models agree" % (agree, responded),
        "models_responded": responded,
        "breakdown": breakdown,
    }


def meta_learner(predictions, strategy="average"):
    if not predictions:
        raise ValueError("No predictions to combine")
    fake_probs = [p["fake_prob"] for p in predictions]
    if strategy == "voting":
        votes_fake = sum(1 for p in predictions if p["verdict"] == "fake")
        final_prob = votes_fake / len(predictions)
    elif strategy == "weighted":
        weights = [MODEL_WEIGHTS.get(p.get("model_id", ""), 1.0) for p in predictions]
        total = sum(weights)
        final_prob = sum(w * p for w, p in zip(weights, fake_probs)) / total if total > 0 else sum(fake_probs) / len(fake_probs)
    else:
        final_prob = sum(fake_probs) / len(fake_probs)
    verdict = "fake" if final_prob >= 0.5 else "real"
    confidence = final_prob if verdict == "fake" else (1 - final_prob)
    return {"verdict": verdict, "fake_probability": round(final_prob, 4), "confidence": round(confidence, 4)}


# -- Routes --

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    media_type: str = Form("image"),
    models: Optional[str] = Form(None),
    strategy: str = Form("weighted"),
):
    if media_type not in MODEL_SERVICES:
        raise HTTPException(400, "media_type must be one of %s" % list(MODEL_SERVICES))

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(400, "Empty file")

    if media_type == "image":
        all_services = MODEL_SERVICES[media_type]
        if models:
            requested = {m.strip() for m in models.split(",")}
            services = [s for s in all_services if s["id"] in requested]
            if not services:
                raise HTTPException(400, "No matching models. Available: %s" % [s["id"] for s in all_services])
        else:
            services = all_services

        async with httpx.AsyncClient() as client:
            tasks = [call_model(client, svc, file_bytes, file.filename or "upload") for svc in services]
            raw_results = await asyncio.gather(*tasks)

        valid = [r for r in raw_results if r["verdict"] != "error"]
        if not valid:
            raise HTTPException(502, "All model services failed.")

        tally = majority_vote(valid)
        token = str(uuid.uuid4())
        verification_id = _gen_verification_id()

        enriched_breakdown = _enrich_breakdown(raw_results)

        sha256 = hashlib.sha256(file_bytes).hexdigest()
        file_type = _file_ext(file.filename or "upload.png")
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        dimensions = "unknown"
        try:
            img = Image.open(io.BytesIO(file_bytes))
            dimensions = "%dx%d" % (img.width, img.height)
        except Exception:
            pass

        verdict_display = "SYNTHETIC (AI)" if tally["verdict"] == "fake" else "AUTHENTIC (REAL)"
        confidence_score = round(sum(r["fake_prob"] for r in valid) / len(valid), 4)

        thumbnail_url = _save_thumbnail(file_bytes, verification_id)

        audit_record = {
            "verification_id": verification_id,
            "feedback_token": token,
            "filename": file.filename,
            "file_type": file_type,
            "dimensions": dimensions,
            "sha256": sha256,
            "timestamp_utc": timestamp,
            "verdict": verdict_display,
            "confidence": confidence_score,
            "model_breakdown": enriched_breakdown,
            "disputed": False,
            "thumbnail_url": thumbnail_url,
        }
        _write_detection(audit_record)

        _prune_detections()
        _detections_cache[token] = {
            "timestamp": time.time(),
            "original_verdict": tally["verdict"],
            "model_breakdown": tally["breakdown"],
        }

        errored = [r["model"] for r in raw_results if r["verdict"] == "error"]
        full_breakdown = list(enriched_breakdown)
        for name in errored:
            display_key, label = MODEL_DISPLAY.get(name, (name, name))
            full_breakdown.append({
                "model": display_key,
                "internal_name": name,
                "label": label,
                "score": None,
                "vote": "error",
                "display_type": "probability",
            })
        order = [s["name"] for s in all_services]
        full_breakdown.sort(key=lambda e: order.index(e["internal_name"]) if e["internal_name"] in order else 99)

        return JSONResponse({
            "verification_id": verification_id,
            "feedback_token": token,
            "receipt": {
                "verdict": verdict_display,
                "confidence_label": tally["confidence_label"],
                "confidence": confidence_score,
                "models_responded": tally["models_responded"],
                "models_total": len(services),
            },
            "model_breakdown": full_breakdown,
            "thumbnail_url": thumbnail_url,
        })

    # Audio / Video
    all_services = MODEL_SERVICES[media_type]
    if models:
        requested = {m.strip() for m in models.split(",")}
        services = [s for s in all_services if s["id"] in requested]
        if not services:
            raise HTTPException(400, "No matching models. Available: %s" % [s["id"] for s in all_services])
    else:
        services = all_services

    async with httpx.AsyncClient() as client:
        tasks = [call_model(client, svc, file_bytes, file.filename or "upload") for svc in services]
        raw_results = await asyncio.gather(*tasks)

    valid = [r for r in raw_results if r["verdict"] != "error"]
    if not valid:
        raise HTTPException(502, "All model services failed.")

    meta = meta_learner(valid, strategy=strategy)
    return JSONResponse({
        "filename": file.filename,
        "media_type": media_type,
        "strategy": strategy,
        "verdict": meta["verdict"],
        "fake_probability": meta["fake_probability"],
        "confidence": meta["confidence"],
        "model_results": list(raw_results),
    })


@app.get("/api/models")
async def list_models():
    return {
        k: [{"id": s["id"], "name": s["name"], "url": s["url"]} for s in v]
        for k, v in MODEL_SERVICES.items()
    }


@app.get("/api/history")
async def history(
    filter: str = Query("all"),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if filter not in ("all", "synthetic", "authentic", "disputed"):
        raise HTTPException(400, "filter must be one of: all, synthetic, authentic, disputed")

    records = _load_all_detections()
    total_all = len(records)

    if filter == "synthetic":
        records = [r for r in records if r.get("verdict") == "SYNTHETIC (AI)"]
    elif filter == "authentic":
        records = [r for r in records if r.get("verdict") == "AUTHENTIC (REAL)"]
    elif filter == "disputed":
        records = [r for r in records if r.get("disputed") is True]

    if search:
        q = search.lower()
        records = [
            r for r in records
            if q in (r.get("filename") or "").lower()
            or q in (r.get("sha256") or "").lower()
            or q in (r.get("verification_id") or "").lower()
        ]

    total_filtered = len(records)
    page = records[offset:offset + limit]

    results = []
    for r in page:
        results.append({
            "verification_id": r.get("verification_id"),
            "filename": r.get("filename"),
            "file_type": r.get("file_type"),
            "dimensions": r.get("dimensions"),
            "thumbnail_url": r.get("thumbnail_url"),
            "timestamp_utc": r.get("timestamp_utc"),
            "verdict": r.get("verdict"),
            "confidence": r.get("confidence"),
            "disputed": r.get("disputed", False),
        })

    return {
        "total_attested": total_all,
        "total_filtered": total_filtered,
        "limit": limit,
        "offset": offset,
        "results": results,
    }


@app.get("/api/detect/{verification_id}")
async def get_detection(verification_id: str):
    rec = _load_detection_by_id(verification_id)
    if rec is None:
        raise HTTPException(404, "Detection not found: %s" % verification_id)
    return rec


@app.post("/api/feedback")
async def submit_feedback(fb: FeedbackIn):
    if fb.user_verdict not in ("correct", "incorrect"):
        raise HTTPException(400, 'user_verdict must be "correct" or "incorrect"')

    _prune_detections()
    snap = _detections_cache.get(fb.feedback_token)

    # Also try loading from JSONL if not in memory
    if snap is None:
        rec = None
        if os.path.exists(DETECTIONS_LOG):
            with open(DETECTIONS_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if r.get("feedback_token") == fb.feedback_token:
                            rec = r
                            break
                    except json.JSONDecodeError:
                        continue
        if rec:
            snap = {"original_verdict": "fake" if "SYNTHETIC" in (rec.get("verdict") or "") else "real", "model_breakdown": rec.get("model_breakdown", [])}

    if snap is None:
        raise HTTPException(404, "Unknown or expired feedback_token")

    # Update disputed flag in audit log
    if fb.user_verdict == "incorrect":
        _update_detection_flag(fb.feedback_token, "disputed", True)

    event = {
        "feedback_token": fb.feedback_token,
        "timestamp": time.time(),
        "original_verdict": snap.get("original_verdict", "unknown"),
        "model_breakdown": snap.get("model_breakdown", []),
        "user_verdict": fb.user_verdict,
    }
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    return {"status": "logged", "feedback_token": fb.feedback_token}
