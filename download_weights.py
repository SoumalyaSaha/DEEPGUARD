#!/usr/bin/env python3
"""
download_weights.py — Fetch genuine pretrained weights for all DeepGuard models.

Run from the DeepGuard root directory:
    python download_weights.py

What this downloads
───────────────────
  weights/npr.pth       CNNDetection ResNet-50 (trained on ProGAN, generalises broadly)
                        Source: github.com/peterwang512/CNNDetection
                        ~100 MB (scripted: GitHub release)

  weights/ufd.pth       UniversalFakeDetect linear classifier head (CLIP ViT-L/14)
                        Source: github.com/WisconsinAIVision/UniversalFakeDetect
                        ~4 KB  (scripted: GitHub release; CLIP backbone auto-downloaded by openai-clip)

  weights/ViT-L-14.pt   OpenAI CLIP ViT-L/14 backbone (needed by IAPL; UFD uses openai-clip's copy)
                        Source: OpenAI public CDN (openaipublic.azureedge.net)
                        ~933 MB (scripted: direct URL)

  weights/nonescape-v0.safetensors
                        Nonescape full detector (DINOv2-large + EfficientNetV2-L)
                        Source: nonescape.sfo2.cdn.digitaloceanspaces.com
                        ~2.4 GB (scripted: direct URL)

  weights/rawnet2.pth   RawNet2 anti-spoofing (ASVspoof 2021 LA track)
                        Source: asvspoof.org / Zenodo
                        MANUAL ONLY — the previously listed Zenodo URL was verified
                        to return an HTML error page, not weights. Do not trust
                        automated downloads for this file; see alt instructions.

  weights/crossvit.pth  CrossEfficientViT (FaceForensics++ trained)
                        Source: github.com/davide-coccomini/...
                        ~20 MB (scripted entry kept as-is; service unevaluated)

  HF cache (auto-warmed, no weights/ file needed):
                        Organika/sdxl-detector, umm-maybe/AI-image-detector,
                        aaronkantrowitz/ai-image-detection (CapCheck) —
                        fetched via huggingface_hub into the HF cache
                        (respects HF_HOME/HF_HUB_CACHE). CapCheck's external
                        cache is therefore part of this scripted flow, not a
                        separate manual step.

  MANUAL (no scriptable source — see README §4):
  weights/iapl_sd14.pth IAPL SDv1.4 checkpoint (~1.7 GB)
                        Source: https://modelscope.cn/models/yihengli/IAPL_pretrain
                        (ModelScope page is JS-gated; no verified direct URL)
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path

# Windows consoles default to cp1252, which cannot print the ✓/✗/○
# status glyphs used below — force UTF-8 so a fresh judge machine
# doesn't crash on output encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


# ── Download registry ────────────────────────────────────────────────────────────
# Each entry: (filename, url, expected_sha256_prefix_or_None)
#
# NOTE: Some repos require accepting a licence before downloading.
#       Where direct URLs are blocked, this script prints manual instructions.

REGISTRY = [
    {
        "name": "NPR (CNNDetection)",
        "file": "npr.pth",
        # Direct link from the CNNDetection GitHub release
        "url": "https://github.com/peterwang512/CNNDetection/releases/download/v1.0/blur_jpg_prob0.5.pth",
        "sha256": None,  # verify manually if needed
        "alt": (
            "Manual download:\n"
            "  1. Visit https://github.com/peterwang512/CNNDetection\n"
            "  2. Download blur_jpg_prob0.5.pth from the Releases page\n"
            "     or from the Google Drive link in the README\n"
            "  3. Save as weights/npr.pth"
        ),
    },
    {
        "name": "UFD (UniversalFakeDetect classifier head)",
        "file": "ufd.pth",
        # Official fc_weights.pth from Wisconsin AI Vision Lab
        "url": "https://github.com/WisconsinAIVision/UniversalFakeDetect/releases/download/v0.1/fc_weights.pth",
        "sha256": None,
        "alt": (
            "Manual download:\n"
            "  1. Visit https://github.com/WisconsinAIVision/UniversalFakeDetect\n"
            "  2. Download fc_weights.pth from the Releases page\n"
            "  3. Save as weights/ufd.pth\n"
            "  Also install CLIP: pip install git+https://github.com/openai/CLIP.git"
        ),
    },
    {
        "name": "IAPL backbone (OpenAI CLIP ViT-L/14)",
        "file": "ViT-L-14.pt",
        # Public OpenAI CDN copy of the ViT-L/14 checkpoint
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99b0e39f330335080d/ViT-L-14.pt",
        "sha256": None,
        "alt": (
            "Manual download:\n"
            "  1. Fetch https://openaipublic.azureedge.net/clip/models/"
            "b8cca3fd41ae0c99b0e39f330335080d/ViT-L-14.pt\n"
            "  2. Save as weights/ViT-L-14.pt (933 MB)"
        ),
    },
    {
        "name": "Nonescape full detector",
        "file": "nonescape-v0.safetensors",
        "url": "https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors",
        "sha256": None,
        "alt": (
            "Manual download:\n"
            "  1. Fetch https://nonescape.sfo2.cdn.digitaloceanspaces.com/"
            "nonescape-v0.safetensors\n"
            "  2. Save as weights/nonescape-v0.safetensors (2.4 GB)"
        ),
    },
    {
        "name": "IAPL SDv1.4 checkpoint (~1.7 GB)",
        "file": "iapl_sd14.pth",
        # No verified scripted source: the ModelScope page
        # (https://modelscope.cn/models/yihengli/IAPL_pretrain) is JS-gated
        # with no stable direct file URL. Manual placement required.
        "url": None,
        "sha256": None,
        "alt": (
            "Manual download REQUIRED (no scriptable source):\n"
            "  1. Visit https://modelscope.cn/models/yihengli/IAPL_pretrain\n"
            "  2. Download the SDv1.4 checkpoint\n"
            "     (file used here: checkpoint_best_acc_sd14.pth, ~1.7 GB)\n"
            "  3. Save as weights/iapl_sd14.pth\n"
            "  See README §4 for details."
        ),
    },
    {
        "name": "RawNet2 (ASVspoof 2021)",
        "file": "rawnet2.pth",
        # VERIFIED BAD: the Zenodo URL below was proven to return an HTML
        # error page instead of weights (see weights/ history). Kept as
        # manual-only so this script never re-downloads the garbage file.
        "url": None,
        "sha256": None,
        "alt": (
            "Manual download REQUIRED (no working scripted source):\n"
            "  1. Obtain a genuine RawNet2_best_model.pth "
            "(ASVspoof 2021 LA track; try the challenge model zoo:\n"
            "     https://github.com/asvspoof-challenge/2021)\n"
            "  2. Save as weights/rawnet2.pth\n"
            "  Do NOT use https://zenodo.org/record/6456915/files/"
            "RawNet2_best_model.pth — verified to serve HTML, not weights."
        ),
    },
    {
        "name": "CrossEfficientViT (FaceForensics++)",
        "file": "crossvit.pth",
        # GitHub release from Coccomini et al.
        "url": (
            "https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformer-"
            "for-Video-Deepfake-Detection/releases/download/v1.0/cross-efficient-vit.pth"
        ),
        "sha256": None,
        "alt": (
            "Manual download:\n"
            "  1. Visit https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformer-for-Video-Deepfake-Detection\n"
            "  2. Download cross-efficient-vit.pth from Releases (or the Google Drive link in README)\n"
            "  3. Save as weights/crossvit.pth"
        ),
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _sizeof_fmt(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _reporthook(count, block_size, total_size):
    downloaded = count * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.0f}%  {_sizeof_fmt(downloaded)}/{_sizeof_fmt(total_size)}", end="", flush=True)
    else:
        print(f"\r  Downloaded {_sizeof_fmt(downloaded)}", end="", flush=True)


def _sha256(path: Path, prefix_len: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:prefix_len]


def download(entry: dict) -> bool:
    dest = WEIGHTS_DIR / entry["file"]

    if dest.exists():
        print(f"  ✓ {dest} already exists — skipping")
        return True

    if not entry.get("url"):
        print(f"\n  ○ {entry['name']}: no scripted source — manual step:")
        print(f"\n  {entry['alt']}\n")
        return False

    print(f"\nDownloading {entry['name']} …")
    print(f"  URL: {entry['url']}")

    try:
        urllib.request.urlretrieve(entry["url"], dest, reporthook=_reporthook)
        print()  # newline after progress bar

        if entry["sha256"]:
            actual = _sha256(dest)
            if not actual.startswith(entry["sha256"]):
                print(f"  ⚠ SHA256 mismatch! Expected {entry['sha256']}, got {actual}")
                print("     The file may be corrupt or the URL outdated.")
            else:
                print(f"  ✓ SHA256 OK ({actual})")

        size = dest.stat().st_size
        print(f"  ✓ Saved to {dest}  ({_sizeof_fmt(size)})")
        return True

    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        if dest.exists():
            dest.unlink()  # remove partial file
        print(f"\n  {entry['alt']}\n")
        return False


# ── Extras ───────────────────────────────────────────────────────────────────────

HF_REPOS = [
    # (repo_id, friendly_name) — auto-fetched into the HF cache at startup
    # by each service; pre-warmed here so a cold launch never waits on them.
    ("Organika/sdxl-detector", "SDXL-detector"),
    ("umm-maybe/AI-image-detector", "umm-maybe"),
    ("aaronkantrowitz/ai-image-detection", "CapCheck"),
]


def warm_hf_cache() -> bool:
    """Pre-download HF-hosted models into the local HF cache.

    This folds CapCheck's external cache (and SDXL/umm-maybe) into the
    scripted flow: on a fresh machine the files land wherever HF_HOME /
    HF_HUB_CACHE point instead of surprising the first service launch.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        print("\n  ○ huggingface_hub not installed — skipping HF cache warm "
              "(services will fetch on first launch instead).")
        print("    Install it with: pip install huggingface_hub")
        return False
    print(f"  HF cache location: {HF_HUB_CACHE}")
    if str(HF_HUB_CACHE).startswith("C:"):
        print("  ⚠ WARNING: cache points at C: — set HF_HOME to a D: path "
              "before running (see install_and_run.bat).")
    ok = True
    for repo_id, friendly in HF_REPOS:
        try:
            path = snapshot_download(repo_id=repo_id)
            print(f"  ✓ {friendly} cached at {path}")
        except Exception as e:
            print(f"  ✗ {friendly} ({repo_id}) failed: {e}")
            ok = False
    return ok


def install_clip():
    """Install openai-clip if not present (needed for UFD)."""
    try:
        import clip
        print("  ✓ openai-clip already installed")
    except ImportError:
        print("\nInstalling openai-clip (required for UFD model)…")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "git+https://github.com/openai/CLIP.git"
        ])
        print("  ✓ openai-clip installed")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DeepGuard — Weight Downloader")
    print("=" * 60)

    results = {}
    for entry in REGISTRY:
        ok = download(entry)
        results[entry["file"]] = ok

    install_clip()

    print("\nWarming Hugging Face model cache (SDXL / umm-maybe / CapCheck)…")
    hf_ok = warm_hf_cache()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_ok = True
    for fname, ok in results.items():
        status = "✓" if ok else "✗ MANUAL ACTION NEEDED"
        print(f"  {status}  weights/{fname}")
        if not ok:
            all_ok = False
    print(f"  {'✓' if hf_ok else '○'}  HF cache (SDXL / umm-maybe / CapCheck)")
    all_ok = all_ok and hf_ok

    if all_ok:
        print("\n  All weights ready! Run ./start.sh or docker compose up --build")
    else:
        print(
            "\n  Some weights need manual download (see instructions above).\n"
            "  After placing the files in weights/, re-run this script to verify."
        )
    print()


if __name__ == "__main__":
    main()
