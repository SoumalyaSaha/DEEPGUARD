"""
Stream TheKernel01/AIGC-Detection-Benchmark (test, 125k rows) and write a
stratified local sample to test_data/hf_benchmark/ without downloading
the full 32GB: images are fetched only for rows kept under quota.

Quotas: 100 real (label==0, generator==0) + 100 fake (label==1),
  including >=20 SDXL (generator==11), rest spread over other fake
  generator classes first-come-first-served.
Filenames encode generator: fake_11_SDXL_0007.jpg / real_0003.jpg
"""
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

GEN_NAMES = {
    0: "Real", 1: "ADM", 2: "BigGAN", 3: "CycleGAN", 4: "DALLE2",
    5: "GauGAN", 6: "GLIDE", 7: "Midjourney", 8: "ProGAN", 9: "SD14",
    10: "SD15", 11: "SDXL", 12: "StarGAN", 13: "StyleGAN",
    14: "StyleGAN2", 15: "VQDM", 16: "WhichFaceIsReal", 17: "Wukong",
}

N_REAL = 100
N_FAKE = 100
N_SDXL_MIN = 20
PER_CLASS_CAP = 8  # spread the non-SDXL fake quota across classes
MAX_SCAN = 60000  # safety cap on streamed rows

OUT = os.path.join("test_data", "hf_benchmark")
os.makedirs(OUT, exist_ok=True)


def main():
    ds = load_dataset("TheKernel01/AIGC-Detection-Benchmark",
                      split="test", streaming=True)
    real_n = 0
    fake_counts = {}
    fake_total = 0
    scanned = 0
    for row in ds:
        scanned += 1
        label = int(row["label"])
        gen = int(row["generator"])
        if label == 0 and gen == 0 and real_n < N_REAL:
            row["image"].convert("RGB").save(
                os.path.join(OUT, f"real_{real_n:04d}.jpg"))
            real_n += 1
        elif label == 1 and fake_total < N_FAKE:
            c = fake_counts.get(gen, 0)
            if gen == 11 or c < PER_CLASS_CAP:
                row["image"].convert("RGB").save(
                    os.path.join(OUT,
                                 f"fake_{gen:02d}_{GEN_NAMES.get(gen, 'UNK')}_{c:04d}.jpg"))
                fake_counts[gen] = c + 1
                fake_total += 1
        if real_n >= N_REAL and fake_total >= N_FAKE \
                and fake_counts.get(11, 0) >= N_SDXL_MIN:
            break
        if scanned >= MAX_SCAN:
            print("HIT MAX_SCAN cap", flush=True)
            break
    print(f"scanned rows: {scanned}", flush=True)
    print(f"real: {real_n}/{N_REAL}", flush=True)
    print(f"fake total: {fake_total}/{N_FAKE} (SDXL: {fake_counts.get(11, 0)})",
          flush=True)
    for g in sorted(fake_counts):
        print(f"  generator {g:02d} {GEN_NAMES.get(g, 'UNK'):15s}: {fake_counts[g]}",
              flush=True)


if __name__ == "__main__":
    main()
