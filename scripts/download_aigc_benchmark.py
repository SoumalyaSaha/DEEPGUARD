"""Download a balanced subset: ~100 images per generator class."""
import os, io
os.environ["HF_HOME"] = r"D:\DeepGuard\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"D:\DeepGuard\hf_cache\datasets"

from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image

FAKE_DIR = Path(r"D:\dataset\HF_FAKE")
REAL_DIR = Path(r"D:\dataset\HF_REAL")
FAKE_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)

REPO = "TheKernel01/AIGC-Detection-Benchmark"
PER_CLASS = 100  # images per generator

gen_names = ["Real","ADM","BigGAN","CycleGAN","DALLE2","GauGAN","GLIDE",
             "Midjourney","ProGAN","SD14","SD15","SDXL","StarGAN",
             "StyleGAN","StyleGAN2","VQDM","WhichFaceIsReal","Wukong"]

counts = {g: 0 for g in gen_names}
fake_count = 0
real_count = 0
total = 0
done = False

for shard in range(60):
    if done:
        break
    fname = f"data/test-{shard:05d}-of-00060.parquet"
    print(f"Shard {shard}/59...", end=" ", flush=True)

    local = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset")
    df = pd.read_parquet(local)

    for i, row in df.iterrows():
        gen_id = int(row["generator"])
        gen_name = gen_names[gen_id] if gen_id < len(gen_names) else str(gen_id)

        if counts[gen_name] >= PER_CLASS:
            continue

        label = row["label"]
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data.get("bytes", b"")
        elif isinstance(img_data, bytes):
            img_bytes = img_data
        else:
            continue
        if not img_bytes:
            continue

        img = Image.open(io.BytesIO(img_bytes))
        stem = f"{gen_name}_{shard:02d}_{total:05d}"

        if label == 1:
            img.save(FAKE_DIR / f"{stem}.png")
            fake_count += 1
        else:
            img.save(REAL_DIR / f"{stem}.png")
            real_count += 1

        counts[gen_name] += 1
        total += 1

    os.remove(local)
    print(f"done (total={total})")

    if all(c >= PER_CLASS for c in counts.values()):
        done = True

print(f"\n=== Done ===")
print(f"Total: {total} images")
print(f"Real:  {real_count} -> {REAL_DIR}")
print(f"Fake:  {fake_count} -> {FAKE_DIR}")
print(f"\nPer generator:")
for g, c in counts.items():
    print(f"  {g:20s}: {c}")
