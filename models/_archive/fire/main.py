"""
FIRE -- Frequency-guIded Reconstruction Error diffusion-fake detector.
Upstream: https://github.com/Chuchad/FIRE (MIT, CVPR 2025).
Model class vendored verbatim in fire_vendor.py (origin: utils/network_utils.py).

Checkpoint: fire_imagenet_adm.pth (ImageNet+ADM, general-purpose per plan).

Preprocessing replicates upstream InversionDataset for mode="fire"
(dataset.py): PIL RGB convert -> ToTensor ONLY (no Resize/Crop/Normalize;
their eval default is resize=False on natively-256px data).

Inference mirrors upstream eval.py: out = model(im)[0]; prob = sigmoid(out).
"""
import io, os, sys, time, logging
import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Keep HuggingFace + torch-hub downloads (SD-v1.5 VAE, ResNet weights) in the
# in-repo/on-D: caches (C: is nearly full). Archived with the service.
os.environ.setdefault("HF_HOME", r"D:\DeepGuard\hf_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\DeepGuard\hf_cache\hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\DeepGuard\hf_cache\hub")
os.environ.setdefault("TORCH_HOME", r"D:\temp\torch_cache")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fire_vendor import FIRE_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fire")
app = FastAPI(title="FIRE Diffusion Deepfake Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "../../weights/fire_imagenet_adm.pth")

model = None
CKPT_MATCHED = 0
CKPT_TOTAL = 0


# Upstream InversionDataset (mode="fire"): RGB -> ToTensor, nothing else
# (their eval default is resize=False on natively-256px data).
# Two mechanical additions for phone photos (up to 4160px here):
#  1. Resize(256) short-edge, exactly the upstream `resize` flag's transform:
#     matches the 256px training distribution AND avoids CUDA OOM in the VAE
#     (a 13MP input tried to allocate ~3GB transient on a 6GB card).
#  2. Reflect-pad H/W up to a multiple of 8 (VAE rounds dims down to mult-of-8).
# Padding/resize are invisible to the classifier (avgpool + single logit).
def _pad_to_mult8(t):
    import torch.nn.functional as F
    _, h, w = t.shape
    return F.pad(t, (0, (8 - w % 8) % 8, 0, (8 - h % 8) % 8), mode="reflect")


TRANSFORM = T.Compose([
    T.Resize(256, antialias=True),
    T.ToTensor(),
    T.Lambda(_pad_to_mult8),
])


@app.on_event("startup")
async def load_model():
    global model, CKPT_MATCHED, CKPT_TOTAL
    logger.info(f"Building FIRE_model (upstream defaults: frq backend, instance norm) on {DEVICE}")
    logger.info(f"HF_HOME={os.environ.get('HF_HOME')}")
    model = FIRE_model()
    if DEVICE.type == "cpu":
        model = model.to("cpu").float()

    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(f"FIRE checkpoint not found at {WEIGHTS_PATH}.")

    logger.info(f"Loading FIRE checkpoint from {WEIGHTS_PATH} (strict=True)")
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    CKPT_TOTAL = len(state)
    try:
        missing, unexpected = model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"FIRE strict checkpoint load FAILED: {e}")
    if missing or unexpected:
        raise RuntimeError(
            f"FIRE strict checkpoint load FAILED: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    CKPT_MATCHED = len(state)
    logger.info(f"FIRE checkpoint loaded (strict=True): {CKPT_MATCHED}/{CKPT_TOTAL} keys matched")

    model = model.to(DEVICE)
    model.eval()
    logger.info(f"FIRE ready on {DEVICE}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "FIRE",
        "variant": "imagenet+adm",
        "device": str(DEVICE),
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
        "ckpt_keys_matched": f"{CKPT_MATCHED}/{CKPT_TOTAL}",
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(tensor)[0]
            fake_prob = torch.sigmoid(out).item()
        return {
            "model": "FIRE",
            "fake_probability": round(fake_prob, 4),
            "verdict": "fake" if fake_prob > 0.5 else "real",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        logger.error(f"FIRE detect error: {e}")
        raise HTTPException(500, detail=str(e))
