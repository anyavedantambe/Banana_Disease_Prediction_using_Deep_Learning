"""How well does a model trained on the Tanzania data work on ANOTHER dataset's matching classes?

    python cross_dataset_eval.py --run_dir /content/drive/MyDrive/banana_results/vgg16_blocks100 \
        --manifest manifests/banana_leaf.csv --data_root /content/data/banana_leaf

Uses only manifest rows that are kept (no duplicates/conflicts), are on the chosen organ (default
leaf), and have a matching Tanzania class (see TANZANIA_MAP in build_manifest.py: these matches are
ASSUMPTIONS). The model never trains on these images, so every image is test data.

Read the results per class, not just overall: the class mix here differs from the Tanzania test set,
so overall accuracy is not comparable with your in-domain number, but per-class recall is.
Writes to <run_dir>/cross_dataset/.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from tensorflow import keras

import data

DEFAULT_CLASSES = "black_sigatoka,fusarium_wilt,healthy"  # the 3 Tanzania classes, in training order


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def in_domain_macro_f1(run_dir):
    p = run_dir / "report.txt"
    if p.exists():
        m = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", p.read_text())
        if m:
            return float(m.group(3))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="trained Tanzania run (model.keras, history.json)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", required=True, help="folder the manifest's rel_path values are relative to")
    ap.add_argument("--organ", default="leaf", help="which organ to include (leaf, fruit, or 'all')")
    ap.add_argument("--classes", default=None, help="model's classes in training order (default: from the run)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    saved = {}
    if (run_dir / "history.json").exists():
        saved = json.loads((run_dir / "history.json").read_text()).get("args", {})
    classes = args.classes or saved.get("classes") or DEFAULT_CLASSES
    class_names = classes.split(",") if isinstance(classes, str) else list(classes)
    backbone = saved.get("backbone", "vgg16")
    img_size = saved.get("img_size", 224)

    model = keras.models.load_model(run_dir / "model.keras")
    if model.output_shape[-1] != len(class_names):
        raise SystemExit(f"Model has {model.output_shape[-1]} outputs but classes are {class_names}. Pass --classes.")

    m = pd.read_csv(args.manifest)
    m = m[m["keep"] & m["tanzania_label"].notna() & (m["tanzania_label"] != "")]
    if args.organ != "all":
        m = m[m["organ"] == args.organ]
    unmatched = sorted(set(m["tanzania_label"]) - set(class_names))
    if unmatched:
        print(f"WARNING: skipping classes the model was not trained on: {unmatched}")
        m = m[m["tanzania_label"].isin(class_names)]
    if m.empty:
        raise SystemExit("No images left to evaluate (check --organ, the manifest, and the model's classes).")
    root = Path(args.data_root)
    df = pd.DataFrame({"rel_path": m["rel_path"].values, "label": m["tanzania_label"].values,
                       "native": m["label"].values, "source": m["source"].values, "group": m["group"].values})
    df["path"] = [str(root / r) for r in df["rel_path"]]
    missing = [p for p in df["path"] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images not found under {root} (e.g. {missing[0]}). Check --data_root.")

    ds = data.make_dataset(df, class_names, img_size=img_size, batch_size=args.batch_size,
                           training=False, backbone=backbone)
    probs = softmax(model.predict(ds, verbose=0))
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    idx = {c: i for i, c in enumerate(class_names)}
    y = df["label"].map(idx).values
    present = sorted(set(y))

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"Model: {run_dir.name} (backbone {backbone}); data: {args.manifest} ({len(df)} images, organ={args.organ})")
    log("Classes evaluated (Tanzania label <- native label, n): " + "; ".join(
        f"{t} <- {n} ({c})" for (t, n), c in df.groupby(["label", "native"]).size().items()))

    cm = pd.DataFrame(confusion_matrix(y, pred, labels=range(len(class_names))),
                      index=[f"true {c}" for c in class_names], columns=[f"pred {c}" for c in class_names])
    log("\nConfusion matrix (rows = assumed true Tanzania class):\n" + cm.to_string())

    p, r, f, s = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
    log(f"\n{'class':<18}{'n':>6}{'recall':>9}{'precision':>11}{'f1':>7}{'mean conf.':>12}")
    for i, c in enumerate(present):
        log(f"{class_names[c]:<18}{s[i]:>6}{r[i]:>9.3f}{p[i]:>11.3f}{f[i]:>7.3f}{conf[y == c].mean():>12.2f}")
    macro = f1_score(y, pred, labels=present, average="macro", zero_division=0)
    acc = float((pred == y).mean())
    log(f"\nAccuracy {acc:.3f} | macro-F1 over the classes present {macro:.3f}")
    ref = in_domain_macro_f1(run_dir)
    if ref is not None:
        log(f"In-domain macro-F1 on the Tanzania test set (from this run's report.txt): {ref:.3f}")
        log("Compare per-class recall, not overall accuracy: the class mix differs between the two test sets.")

    log("\nBy native class: share of images the model calls each Tanzania class")
    native_rows = []
    for native, g in df.groupby("native"):
        shares = {class_names[c]: float((pred[g.index] == c).mean()) for c in range(len(class_names))}
        log(f"  {native} (n={len(g)}, assumed {g['label'].iloc[0]}): " +
            ", ".join(f"{k} {100 * v:.0f}%" for k, v in shares.items()) +
            f" | mean confidence {conf[g.index].mean():.2f}")
        native_rows.append({"native": native, "n": len(g), **shares})
    log("\nCaution: the Tanzania model never saw these backgrounds, cameras or dates, and the class matches are "
        "assumptions (see TANZANIA_MAP). A drop here mixes domain shift with label mismatch.")

    out = run_dir / "cross_dataset"
    out.mkdir(exist_ok=True)
    stem = Path(args.manifest).stem
    per = df[["rel_path", "native", "label", "source", "group"]].copy()
    per["pred"] = [class_names[i] for i in pred]
    per["confidence"] = conf
    for i, c in enumerate(class_names):
        per[f"p_{c}"] = probs[:, i]
    per.to_csv(out / f"{stem}_predictions.csv", index=False)
    pd.DataFrame(native_rows).to_csv(out / f"{stem}_by_native_class.csv", index=False)
    (out / f"{stem}_summary.txt").write_text("\n".join(lines))
    log(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
