"""
NPR -- Noise Pattern Recognition image deepfake detector (CNNDetection ResNet-50).
Upstream CNNDetection convention (PeterWang512/CNNDetection): the model's sigmoid
output IS the probability of being synthetic/fake ("probability of being synthetic").
No inversion. Weight file: weights/npr.pth
"""
import io, time, logging, os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("npr")
app = FastAPI(title="NPR Image Deepfake Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "../../weights/npr.pth")


class NPRModel(nn.Module):
    def __init__(self, base=None):
        super().__init__()
        if base is None:
            base = resnet50(weights=None)
            base.fc = nn.Linear(2048, 1)
        self.net = base

    def forward(self, x):
        return torch.sigmoid(self.net(x))


model = None
# Upstream CNNDetection demo.py pipeline: CenterCrop(224) (no Resize) + ToTensor + ImageNet norm.
TRANSFORM = T.Compose([
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.on_event("startup")
async def load_model():
    global model
    # Load the checkpoint directly into the bare ResNet50 (checkpoint keys have no
    # "net." prefix) so strict=True verifies every key actually matched.
    base = resnet50(weights=None)
    base.fc = nn.Linear(2048, 1)
    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Loading NPR weights from {WEIGHTS_PATH}")
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = base.load_state_dict(state, strict=True)
        matched = len(state) - len(missing) - len(unexpected)
        logger.info(
            f"NPR checkpoint loaded: {matched}/{len(state)} keys matched "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        logger.warning(f"No weights at {WEIGHTS_PATH}")
    model = NPRModel(base=base).to(DEVICE)
    model.eval()
    logger.info(f"NPR ready on {DEVICE}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "NPR",
        "device": str(DEVICE),
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw = model(tensor).item()
        # Upstream CNNDetection: sigmoid output IS P(fake) ("probability of being synthetic"),
        # so no inversion -- fake_prob = raw.
        fake_prob = raw
        return {
            "model": "NPR",
            "fake_probability": round(fake_prob, 4),
            "verdict": "fake" if fake_prob > 0.5 else "real",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))
