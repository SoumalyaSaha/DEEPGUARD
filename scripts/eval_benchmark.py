"""Run test_data/hf_benchmark/ through one model service, merge per-file
scores into results/benchmark_scores.json (keyed by filename, per-model
sub-objects), and print that model's overall + per-generator accuracy
and score stats. Run once per model, sequentially:
  python scripts/eval_benchmark.py http://localhost:5009/detect sdxl
  python scripts/eval_benchmark.py http://localhost:5010/detect umm
"""
import glob
import json
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

GEN_NAMES = {
    "00": "Real", "01": "ADM", "02": "BigGAN", "03": "CycleGAN",
    "04": "DALLE2", "05": "GauGAN", "06": "GLIDE", "07": "Midjourney",
    "08": "ProGAN", "09": "SD14", "10": "SD15", "11": "SDXL",
    "12": "StarGAN", "13": "StyleGAN", "14": "StyleGAN2", "15": "VQDM",
    "16": "WhichFaceIsReal", "17": "Wukong",
}

URL, TAG = sys.argv[1], sys.argv[2]
OUT = "results/benchmark_scores.json"
try:
    with open(OUT) as f:
        all_scores = json.load(f)
except FileNotFoundError:
    all_scores = {}

files = sorted(glob.glob(os.path.join("test_data", "hf_benchmark", "*")))
print(f"{TAG}: {len(files)} files -> {URL}", flush=True)
for fpath in files:
    fn = os.path.basename(fpath)
    gen = "00" if fn.startswith("real_") else fn.split("_")[1]
    label = 0 if fn.startswith("real_") else 1
    try:
        with open(fpath, "rb") as fh:
            r = httpx.post(URL, files={"file": (fn, fh, "application/octet-stream")},
                           timeout=180.0).json()
        entry = all_scores.setdefault(fn, {"gen": gen, "label": label})
        entry[TAG] = {"p": r["fake_probability"], "verdict": r["verdict"],
                      "latency_ms": r.get("latency_ms")}
    except Exception as e:
        print(f"{fn}: ERROR {str(e)[:100]}", flush=True)

with open(OUT, "w") as f:
    json.dump(all_scores, f)

# summary for this model
groups = {}
for fn, v in all_scores.items():
    if TAG not in v:
        continue
    groups.setdefault((v["label"], v["gen"]), []).append(v[TAG]["p"])
for (label, gen) in sorted(groups):
    ps = groups[(label, gen)]
    acc = sum(1 for p in ps if (p >= 0.5) == bool(label)) / len(ps)
    print(f"{TAG} label={label} gen={gen} {GEN_NAMES.get(gen, '?'):15s} "
          f"n={len(ps):3d} acc={acc:.3f} mean={sum(ps)/len(ps):.4f} "
          f"min={min(ps):.4f} max={max(ps):.4f}", flush=True)
