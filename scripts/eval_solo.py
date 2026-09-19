"""
scripts/eval_solo.py — Solo-model batch accuracy evaluation.

Usage:
    python scripts/eval_solo.py <folder> <ground_truth_label>
    python scripts/eval_solo.py <folder> <ground_truth_label> <model_url>

Example:
    python scripts/eval_solo.py test_data/real real
    python scripts/eval_solo.py test_data/fake fake
    python scripts/eval_solo.py test_data/real real http://localhost:5005/detect

Calls NPR (:5001) and UFD (:5004) directly — bypasses the gateway.
If <model_url> is given, only that model is evaluated and
results/eval_<label>_<model>.csv is written with model-specific columns
(<model> derived from the URL port, e.g. iapl/sdxl/fire).
Writes results/eval_real.csv (or eval_fake.csv) when no model_url is given.
"""

import os
import re
import sys
import csv
import glob
import httpx

NPR_URL = "http://localhost:5001/detect"
UFD_URL = "http://localhost:5004/detect"
# optional model URL overrides both NPR and UFD (single-model eval)
MODEL_URL = None
if len(sys.argv) == 4:
    MODEL_URL = sys.argv[3].rstrip("/")
# Per-model tag so output CSVs never collide across model runs
# (previously every MODEL_URL run overwrote eval_<label>_iapl.csv).
PORT_MODEL = {"5001": "npr", "5004": "ufd", "5005": "iapl", "5006": "fire",
              "5007": "fft", "5009": "sdxl", "5010": "umm", "5011": "capcheck",
              "5013": "nonescape"}
MODEL_TAG = "custom"
if MODEL_URL:
    _m = re.search(r":(\d+)", MODEL_URL)
    MODEL_TAG = PORT_MODEL.get(_m.group(1), "custom") if _m else "custom"
TIMEOUT = httpx.Timeout(300.0)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def call_model(url: str, file_path: str) -> dict:
    try:
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "application/octet-stream")}
            resp = httpx.post(url, files=files, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return {
            "fake_prob": float(data.get("fake_probability", 0.5)),
            "verdict": data.get("verdict", "unknown"),
            "latency_ms": data.get("latency_ms", None),
            "error": None,
        }
    except Exception as e:
        return {
            "fake_prob": 0.5,
            "verdict": "error",
            "latency_ms": None,
            "error": str(e),
        }


def main():
    if len(sys.argv) not in (3, 4):
        print(f"Usage: {sys.argv[0]} <folder> <ground_truth_label> [model_url]")
        print(f"  ground_truth_label = 'real' or 'fake'")
        print(f"  model_url          = optional HTTP endpoint (e.g. http://localhost:5005/detect)")
        sys.exit(1)

    folder = sys.argv[1]
    ground_truth = sys.argv[2].lower()

    if ground_truth not in ("real", "fake"):
        print(f"Error: ground_truth must be 'real' or 'fake', got '{ground_truth}'")
        sys.exit(1)

    if not os.path.isdir(folder):
        print(f"Error: folder '{folder}' not found")
        sys.exit(1)

    # Collect image files
    files = [
        p for p in glob.glob(os.path.join(folder, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    ]
    files.sort()

    if not files:
        print(f"No image files found in {folder}")
        sys.exit(1)

    total = len(files)
    print(f"Evaluating {total} images from {folder} (ground truth: {ground_truth})")
    if MODEL_URL:
        print(f"Model: {MODEL_URL}")
    else:
        print(f"NPR: {NPR_URL}  |  UFD: {UFD_URL}")
    print("-" * 70)

    # Ensure results/ exists
    os.makedirs("results", exist_ok=True)
    label_tag = "real" if ground_truth == "real" else "fake"
    if MODEL_URL:
        csv_path = f"results/eval_{label_tag}_{MODEL_TAG}.csv"
    else:
        csv_path = f"results/eval_{label_tag}.csv"

    rows = []
    npr_correct = 0
    npr_fp = 0   # false positive: ground_truth=real but NPR said fake
    npr_fn = 0   # false negative: ground_truth=fake but NPR said real
    ufd_correct = 0
    ufd_fp = 0
    ufd_fn = 0
    iapl_correct = 0
    iapl_fp = 0   # false positive: ground_truth=real but IAPL said fake
    iapl_fn = 0   # false negative: ground_truth=fake but IAPL said real
    iapl_eval = bool(MODEL_URL)

    for i, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        print(f"[{i:2d}/{total}] {fname} ... ", end="", flush=True)

        if MODEL_URL:
            res = call_model(MODEL_URL, fpath)
            ok = res["verdict"] == ground_truth
            if ok:
                iapl_correct += 1
            elif res["verdict"] == "fake":
                iapl_fp += 1
            else:
                iapl_fn += 1

            mark = "OK" if ok else "!!"
            print(f"{MODEL_TAG.upper()} {mark} (p={res['fake_prob']:.4f})")

            rows.append({
                "filename": fname,
                "iapl_fake_prob": round(res["fake_prob"], 4),
                "iapl_verdict": res["verdict"],
                "iapl_latency_ms": res["latency_ms"],
                "iapl_error": res["error"] or "",
                "ground_truth": ground_truth,
            })
        else:
            npr = call_model(NPR_URL, fpath)
            ufd = call_model(UFD_URL, fpath)

            npr_ok = npr["verdict"] == ground_truth
            ufd_ok = ufd["verdict"] == ground_truth
            if npr_ok:
                npr_correct += 1
            else:
                if npr["verdict"] == "fake":
                    npr_fp += 1
                else:
                    npr_fn += 1
            if ufd_ok:
                ufd_correct += 1
            else:
                if ufd["verdict"] == "fake":
                    ufd_fp += 1
                else:
                    ufd_fn += 1

            # Status line
            npr_mark = "OK" if npr_ok else "!!"
            ufd_mark = "OK" if ufd_ok else "!!"
            print(f"NPR {npr_mark} (p={npr['fake_prob']:.4f})  UFD {ufd_mark} (p={ufd['fake_prob']:.4f})")

            rows.append({
                "filename": fname,
                "npr_fake_prob": round(npr["fake_prob"], 4),
                "npr_verdict": npr["verdict"],
                "npr_latency_ms": npr["latency_ms"],
                "npr_error": npr["error"] or "",
                "ufd_fake_prob": round(ufd["fake_prob"], 4),
                "ufd_verdict": ufd["verdict"],
                "ufd_latency_ms": ufd["latency_ms"],
                "ufd_error": ufd["error"] or "",
                "ground_truth": ground_truth,
            })

    # Write CSV
    if MODEL_URL:
        fieldnames = [
            "filename", "iapl_fake_prob", "iapl_verdict",
            "iapl_latency_ms", "iapl_error", "ground_truth",
        ]
    else:
        fieldnames = [
            "filename", "npr_fake_prob", "npr_verdict", "npr_latency_ms", "npr_error",
            "ufd_fake_prob", "ufd_verdict", "ufd_latency_ms", "ufd_error",
            "ground_truth",
        ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"RESULTS  -  {total} images, ground truth: {ground_truth}")
    print("=" * 70)
    print()
    if iapl_eval:
        iapl_acc = iapl_correct / total * 100
        print(f"  {MODEL_TAG.upper()}: {iapl_correct}/{total} correct = {iapl_acc:.1f}%")
        if ground_truth == "real":
            print(f"        True negatives  (real->real): {iapl_correct}")
            print(f"        False positives (real->fake): {iapl_fp}")
        else:
            print(f"        True positives  (fake->fake): {iapl_correct}")
            print(f"        False negatives (fake->real): {iapl_fn}")
    else:
        npr_acc = npr_correct / total * 100
        ufd_acc = ufd_correct / total * 100
        print(f"  NPR:  {npr_correct}/{total} correct = {npr_acc:.1f}%")
        print(f"        False positives (real->fake): {npr_fp}")
        print(f"        False negatives (fake->real): {npr_fn}")
        print()
        print(f"  UFD:  {ufd_correct}/{total} correct = {ufd_acc:.1f}%")
        print(f"        False positives (real->fake): {ufd_fp}")
        print(f"        False negatives (fake->real): {ufd_fn}")
    print()
    print(f"  CSV written to: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
