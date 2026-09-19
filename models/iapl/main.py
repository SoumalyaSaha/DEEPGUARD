"""
IAPL -- Image-Adaptive Prompt Learning image deepfake detector.
Official repo: https://github.com/liyih/IAPL  (Ojha-style CLIP ViT-L/14 + adapters).

Plain single-image inference ONLY in this pass (no test-time adaptation).
Checkpoint: checkpoint_best_acc_sd14.pth (SD v1.4 / GenImage-trained) from
  https://modelscope.cn/models/yihengli/IAPL_pretrain
Backbone architecture built from OpenAI ViT-L-14.pt, then overwritten by the
checkpoint's full state_dict (checkpoint embeds the whole CLIP backbone).

Confirmed config (from run_genimage.sh + main.py defaults):
  backbone=CLIP:ViT-L/14, gate=True, condition=True, n_ctx=2, prompt_depth=9,
  vision_width=1024, image_size=224, vit_adapter_list=[3,7,11,15,19,23].

Preprocessing replicates the GenImage transforms in utils/dataset.py
(test split, GenImage norm stats -- DIFFERENT from UFD/NPR ImageNet stats):
  Resize(256) -> translate_duplicate -> CenterCrop(224) -> ToTensor
  -> Normalize(mean=[0.481,0.458,0.408], std=[0.269,0.261,0.276])
"""
import io, math, os, sys, time, logging
import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# pytorch_wavelets is imported by the vendored package but unused in this path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_wavelets_stub import install_stub
install_stub()

import iapl_vendor
from iapl_vendor import build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iapl")
app = FastAPI(title="IAPL Image Deepfake Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "../../weights/iapl_sd14.pth")
CLIP_PATH = os.getenv("CLIP_PATH", "../../weights/ViT-L-14.pt")

model = None


def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size
        new_width = width * math.ceil(cropSize / width)
        new_height = height * math.ceil(cropSize / height)
        new_img = Image.new("RGB", (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    return img


TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.Lambda(lambda img: translate_duplicate(img, 224)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.481, 0.458, 0.408], [0.269, 0.261, 0.276]),
])

TTA_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.Lambda(lambda img: translate_duplicate(img, 224)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.481, 0.458, 0.408], [0.269, 0.261, 0.276]),
])


def _unpack_logits(outputs):
    """CLIPModel.forward returns [logits, image_features, (pred_bias)] in
    train() mode (see iapl_vendor/clip_models.py + get_criterion which uses
    outputs[0] as logits) and a plain logits tensor in eval() mode."""
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def run_tta(img, n_views=31, micro_batch=4):
    if torch.cuda.is_available():
        logger.info(
            f"TTA start: VRAM allocated={torch.cuda.memory_allocated() / 1024**2:.1f} MB"
        )
    original_ctx = model.prompt_learner.ctx.detach().clone()

    views = [TRANSFORM(img)]
    for _ in range(n_views):
        views.append(TTA_TRANSFORM(img))
    views_tensor = torch.stack(views).to(DEVICE)
    n_total = views_tensor.shape[0]
    logger.info(
        f"TTA: {n_total} views on {views_tensor.device}, "
        f"model on {next(model.parameters()).device}"
    )

    model.freeze_tta()
    model.train()

    optimizer = torch.optim.AdamW([model.prompt_learner.ctx], lr=0.005)

    # Gradient accumulation over micro-batches: a full 32-view batch through
    # ViT-L/14 with backward graph OOMs a 6GB GPU, so accumulate the
    # size-weighted chunk-mean entropy (mathematically identical gradient).
    for _ in range(2):
        optimizer.zero_grad()
        for i in range(0, n_total, micro_batch):
            chunk = views_tensor[i:i + micro_batch]
            logits = _unpack_logits(model(chunk))
            probs = torch.sigmoid(logits)
            entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
            loss = entropy.mean() * (chunk.shape[0] / n_total)
            loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        orig_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        final_logits = _unpack_logits(model(orig_tensor))
    fake_prob = torch.sigmoid(final_logits).item()

    model.prompt_learner.ctx.data.copy_(original_ctx)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(
            f"TTA done: VRAM allocated={torch.cuda.memory_allocated() / 1024**2:.1f} MB"
        )
    return fake_prob


def _make_config():
    """Namespace matching run_genimage.sh + main.py defaults for the SDv1.4 model."""
    class C: pass
    c = C()
    c.backbone = "CLIP:ViT-L/14"
    c.clip_path = CLIP_PATH
    c.image_size = 224
    c.vision_width = 1024
    c.prompt_depth = 9
    c.n_ctx = 2
    c.vit_adapter_list = [3, 7, 11, 15, 19, 23]
    c.text_adapter_list = []
    c.gate = True
    c.condition = True
    c.use_contrast = False
    c.smooth = True
    c.tta = False
    c.loss_adapter = 1.0
    c.loss_contrast = 1.0
    c.loss_condition = 1.0
    return c


@app.on_event("startup")
async def load_model():
    global model
    if not os.path.exists(CLIP_PATH):
        raise RuntimeError(
            f"CLIP backbone file not found at {CLIP_PATH}. "
            "Download OpenAI ViT-L-14.pt (or set CLIP_PATH) before starting."
        )
    cfg = _make_config()
    logger.info(f"Building CLIPModel from {CLIP_PATH} (plain forward, no TTA)")
    model = build_model(cfg)
    model = model.to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Loading IAPL SDv1.4 weights from {WEIGHTS_PATH}")
        ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=True)
        matched = len(state) - len(missing) - len(unexpected)
        logger.info(
            f"IAPL checkpoint loaded (strict=True): {matched}/{len(state)} keys "
            f"matched (missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing or unexpected:
            raise RuntimeError(
                f"IAPL strict checkpoint load FAILED: missing={missing[:10]}, "
                f"unexpected={unexpected[:10]}"
            )
    else:
        logger.warning(
            f"Weight file not found at {WEIGHTS_PATH}. Running with RANDOM "
            "weights -- results meaningless."
        )

    model.eval()
    logger.info(f"IAPL ready on {DEVICE}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "IAPL",
        "variant": "SDv1.4",
        "tta": False,
        "device": str(DEVICE),
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), tta: bool = False):
    t0 = time.time()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        if tta:
            fake_prob = run_tta(img)
        else:
            tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                fake_prob = torch.sigmoid(logits).item()
        return {
            "model": "IAPL",
            "tta": tta,
            "fake_probability": round(fake_prob, 4),
            "verdict": "fake" if fake_prob > 0.5 else "real",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        logger.error(f"IAPL detect error: {e}")
        raise HTTPException(500, detail=str(e))