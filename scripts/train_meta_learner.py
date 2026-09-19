"""
Offline logistic-regression meta-learner for DeepGuard image path.

Merges per-model solo-eval CSVs (NPR+UFD combined, IAPL plain-forward),
trains sklearn LogisticRegression on [npr, ufd, iapl] fake_probs with a
stratified 80/20 split (seed 42), evaluates on the held-out test set, and
saves coefficients to results/meta_learner_weights.json for later gateway use.

Usage:
    venv\\Scripts\\python.exe scripts\\train_meta_learner.py [tricky1.jpg tricky2.jpg ...]
Optional extra image paths are scored with the fitted model (sanity check).
"""
import csv
import json
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

SEED = 42
FEATURES = ["npr", "ufd", "iapl"]


def load_scores():
    """filename -> {npr, ufd, iapl, label}; drops rows with any model error."""
    rows = {}
    for csv_path, col_prefix in [("results/eval_real.csv", "npr"),
                                 ("results/eval_real.csv", "ufd"),
                                 ("results/eval_real_iapl.csv", "iapl")]:
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                fn = r["filename"]
                err = (r.get(f"{col_prefix}_error", "") or "").strip()
                if err:
                    rows[fn] = None
                    continue
                if fn in rows and rows[fn] is None:
                    continue
                rows.setdefault(fn, {"label": 0})[col_prefix] = float(r[f"{col_prefix}_fake_prob"])
    for csv_path, col_prefix in [("results/eval_fake.csv", "npr"),
                                 ("results/eval_fake.csv", "ufd"),
                                 ("results/eval_fake_iapl.csv", "iapl")]:
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                fn = r["filename"]
                err = (r.get(f"{col_prefix}_error", "") or "").strip()
                if err:
                    rows[fn] = None
                    continue
                if fn in rows and rows[fn] is None:
                    continue
                rows.setdefault(fn, {"label": 1})[col_prefix] = float(r[f"{col_prefix}_fake_prob"])

    data = [(fn, v) for fn, v in rows.items()
            if v is not None and all(k in v for k in FEATURES)]
    return data


def main():
    data = load_scores()
    print(f"merged rows: {len(data)} (excluded: "
          f"{50 + 32 - len(data)} missing/errored)")
    X = [[v[f] for f in FEATURES] for _, v in data]
    y = [v["label"] for _, v in data]
    names = [fn for fn, _ in data]

    X_train, X_test, y_train, y_test, n_train, n_test = train_test_split(
        X, y, names, test_size=0.2, random_state=SEED, stratify=y)
    print(f"train: {len(X_train)} "
          f"(real={y_train.count(0)}, fake={y_train.count(1)}), "
          f"test: {len(X_test)} "
          f"(real={y_test.count(0)}, fake={y_test.count(1)}), seed={SEED}")

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    coef = {f: round(float(c), 4) for f, c in zip(FEATURES, clf.coef_[0])}
    intercept = round(float(clf.intercept_[0]), 4)
    print(f"coefficients: {coef}, intercept: {intercept}")

    for split, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = clf.predict(Xs)
        acc = accuracy_score(ys, pred)
        cm = confusion_matrix(ys, pred, labels=[0, 1]).tolist()
        prec, rec, _, _ = precision_recall_fscore_support(
            ys, pred, labels=[0, 1], zero_division=0)
        print(f"{split}: acc={acc:.3f} cm[TN,FP/FN,TP]={cm} "
              f"fake_precision={prec[1]:.3f} fake_recall={rec[1]:.3f}")

    out = {
        "seed": SEED,
        "features": FEATURES,
        "coefficients": coef,
        "intercept": intercept,
        "train_accuracy": round(accuracy_score(y_train, clf.predict(X_train)), 4),
        "test_accuracy": round(accuracy_score(y_test, clf.predict(X_test)), 4),
        "test_files": sorted(n_test),
    }
    with open("results/meta_learner_weights.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/meta_learner_weights.json")

    for img_path in sys.argv[1:]:
        import httpx
        scores = {}
        for mid, url in [("npr", "http://localhost:5001/detect"),
                         ("ufd", "http://localhost:5004/detect"),
                         ("iapl", "http://localhost:5005/detect")]:
            with open(img_path, "rb") as fh:
                r = httpx.post(url, files={"file": ("x.jpg", fh, "application/octet-stream")},
                               timeout=120.0).json()
            scores[mid] = r["fake_probability"]
        x = [[scores[f] for f in FEATURES]]
        p_fake = float(clf.predict_proba(x)[0][1])
        print(f"{img_path}: scores={scores} -> LR p(fake)={p_fake:.4f} "
              f"verdict={'fake' if p_fake >= 0.5 else 'real'}")


if __name__ == "__main__":
    main()
