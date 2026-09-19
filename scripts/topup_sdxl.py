"""Top up SDXL (generator==11) rows to >= 20 in test_data/hf_benchmark/.
Keeps streaming until quota met; images fetched only for kept rows."""
import glob
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

OUT = os.path.join("test_data", "hf_benchmark")
TARGET = 20

existing = sorted(glob.glob(os.path.join(OUT, "fake_11_SDXL_*.jpg")))
have = len(existing)
print(f"existing SDXL: {have}, target: {TARGET}", flush=True)
if have >= TARGET:
    print("already satisfied", flush=True)
    raise SystemExit

ds = load_dataset("TheKernel01/AIGC-Detection-Benchmark",
                  split="test", streaming=True)
n = have
scanned = 0
for row in ds:
    scanned += 1
    if int(row["label"]) == 1 and int(row["generator"]) == 11:
        row["image"].convert("RGB").save(
            os.path.join(OUT, f"fake_11_SDXL_{n:04d}.jpg"))
        n += 1
        print(f"saved SDXL #{n} (scanned {scanned})", flush=True)
        if n >= TARGET:
            break
print(f"done: SDXL={n}, scanned rows: {scanned}", flush=True)
