"""Can a model trained on SOME classes flag images of a class it never saw?

    python ood.py --data_root data/banana_small --run_dir /content/drive/MyDrive/banana_results/holdout_fusarium_s42

Run it on a model trained with --classes (e.g. healthy,black_sigatoka). Every other class found in
the data is the "unknown" (emerging) disease. The known classes, split file and image size are read
from the run's history.json, so you normally only pass --data_root and --run_dir.

Scores (all: HIGHER = more unfamiliar):
  max_softmax      1 - highest softmax probability
  entropy          entropy of the softmax output (only with 3+ known classes: with 2 known classes it
                   ranks images exactly like max_softmax, so it would add nothing)
  class_distance   Euclidean distance, in the 256-d 'features' layer, to the nearest known-class
                   average (class averages come from the TRAINING images of the known classes)

Honest protocol: thresholds are chosen on VALIDATION images of the KNOWN classes only. The unknown
class is never used to choose a threshold. Outputs go to <run_dir>/ood/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tensorflow import keras

import data


# ----------------------------------------------------------------------------
# model outputs and scores
# ----------------------------------------------------------------------------
def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(model, df, class_names, img_size, batch_size):
    """One pass in normal prediction mode (dropout OFF). Returns (features, logits) in df row order."""
    two_out = keras.Model(model.inputs, [model.get_layer("features").output, model.output])
    ds = data.make_dataset(df, class_names, img_size=img_size, batch_size=batch_size, training=False)
    feats, logits = two_out.predict(ds, verbose=0)
    return feats, logits


def compute_scores(feats, logits, centroids, n_known):
    p = softmax(logits)
    scores = {"max_softmax": 1.0 - p.max(axis=1)}
    if n_known >= 3:
        scores["entropy"] = -(p * np.log(p + 1e-12)).sum(axis=1)
    d = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=2)  # (n_images, n_known)
    scores["class_distance"] = d.min(axis=1)
    return scores


def pct(x):
    return f"{100 * x:.1f}%"


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run_dir", required=True, help="folder with model.keras and history.json")
    ap.add_argument("--classes", default=None, help="known classes, in training order (default: from history.json)")
    ap.add_argument("--splits_csv", default=None, help="split file (default: from history.json)")
    ap.add_argument("--unknown", default=None, help="comma-separated unknown classes (default: all non-known)")
    ap.add_argument("--keep", type=float, default=0.95, help="share of known validation images the threshold accepts")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = run_dir / "ood"
    out.mkdir(parents=True, exist_ok=True)
    saved = {}
    if (run_dir / "history.json").exists():
        saved = json.loads((run_dir / "history.json").read_text()).get("args", {})
    classes = args.classes or saved.get("classes")
    if not classes:
        raise SystemExit("Known classes unknown: this run trained on ALL classes (no holdout). "
                         "Pass --classes, or train with --classes first.")
    known = classes.split(",") if isinstance(classes, str) else list(classes)
    splits_csv = args.splits_csv or saved.get("splits_csv") or "splits/splits.csv"
    img_size = args.img_size or saved.get("img_size", 224)
    if not Path(splits_csv).exists():
        raise SystemExit(f"{splits_csv} not found. Pass --splits_csv (the file the run was trained with); "
                         f"this script never creates a new split.")

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    df = data.scan_dataset(args.data_root)
    train_df, val_df, test_df = data.make_splits(df, args.data_root, split_csv=splits_csv)
    unknown = args.unknown.split(",") if args.unknown else sorted(set(df["label"]) - set(known))
    if not unknown:
        raise SystemExit("No unknown classes found in the data.")
    all_names = known + unknown
    log(f"Known classes:   {known}")
    log(f"Unknown classes: {unknown}")
    log(f"Split file: {splits_csv}")

    model = keras.models.load_model(run_dir / "model.keras")
    if model.output_shape[-1] != len(known):
        raise SystemExit(f"Model has {model.output_shape[-1]} outputs but {len(known)} known classes were given. "
                         f"Use the same --classes (and order) as in training.")

    # ---- class averages from TRAINING images of the known classes ----
    tr_known = train_df[train_df["label"].isin(known)].reset_index(drop=True)
    f_tr, _ = forward(model, tr_known, known, img_size, args.batch_size)
    y_tr = tr_known["label"].map({c: i for i, c in enumerate(known)}).values
    centroids = np.stack([f_tr[y_tr == i].mean(axis=0) for i in range(len(known))])

    # ---- thresholds from VALIDATION images of the KNOWN classes only ----
    va_known = val_df[val_df["label"].isin(known)].reset_index(drop=True)
    f_va, l_va = forward(model, va_known, known, img_size, args.batch_size)
    val_scores = compute_scores(f_va, l_va, centroids, len(known))
    thr = {k: float(np.quantile(v, args.keep)) for k, v in val_scores.items()}

    # ---- score the TEST set: known + unknown ----
    te = test_df[test_df["label"].isin(all_names)].reset_index(drop=True)
    f_te, l_te = forward(model, te, all_names, img_size, args.batch_size)
    scores = compute_scores(f_te, l_te, centroids, len(known))
    probs = softmax(l_te)
    is_unk = ~te["label"].isin(known).values
    n_unk, n_kn = int(is_unk.sum()), int((~is_unk).sum())
    log(f"Test images: {n_kn} known, {n_unk} unknown (chance AUPR = {n_unk / len(te):.3f})")
    if "entropy" not in scores:
        log("Note: entropy skipped (2 known classes: it ranks images exactly like max_softmax).")

    # ---- A. the problem: what does the ordinary classifier do with the unknown class? ----
    log("\n=== A. WHAT THE ORDINARY CLASSIFIER DOES WITH UNKNOWN IMAGES (the mislabeling problem) ===")
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    mis_rows = []
    for u in unknown:
        m = (te["label"] == u).values
        row = {"unknown_class": u, "n": int(m.sum()),
               "median_confidence": float(np.median(conf[m])),
               "share_conf_over_0.9": float((conf[m] > 0.9).mean()),
               "share_conf_over_0.99": float((conf[m] > 0.99).mean())}
        for i, k in enumerate(known):
            row[f"labeled_{k}"] = float((pred[m] == i).mean())
        mis_rows.append(row)
        log(f"{u} (n={m.sum()}): labeled " + ", ".join(f"{k} {pct((pred[m] == i).mean())}" for i, k in enumerate(known))
            + f" | median confidence {np.median(conf[m]):.2f}, confidence>0.9 for {pct((conf[m] > 0.9).mean())}")
    pd.DataFrame(mis_rows).to_csv(out / "mislabeling_baseline.csv", index=False)

    # ---- B. detection performance ----
    log(f"\n=== B. DETECTION (threshold keeps {pct(args.keep)} of KNOWN validation images; unknown = positive) ===")
    known_mask = ~is_unk
    y_known_true = te["label"][known_mask].map({c: i for i, c in enumerate(known)}).values
    known_correct = pred[known_mask] == y_known_true
    log(f"Closed-set accuracy on known test images: {pct(known_correct.mean())}")
    results = {}
    header = f"{'score':<16}{'AUROC':>8}{'AUPR':>8}{'unknown flagged':>17}{'known flagged':>15}{'acc. of accepted':>18}"
    log(header)
    for name, s in scores.items():
        flagged = s > thr[name]
        acc_acc = known_correct[~flagged[known_mask]].mean() if (~flagged[known_mask]).any() else float("nan")
        results[name] = {
            "auroc": float(roc_auc_score(is_unk, s)),
            "aupr": float(average_precision_score(is_unk, s)),
            "threshold": thr[name],
            "unknown_flagged": float(flagged[is_unk].mean()),
            "known_false_alarm": float(flagged[known_mask].mean()),
            "accuracy_of_accepted_known": float(acc_acc),
        }
        r = results[name]
        log(f"{name:<16}{r['auroc']:>8.3f}{r['aupr']:>8.3f}{pct(r['unknown_flagged']):>17}"
            f"{pct(r['known_false_alarm']):>15}{pct(r['accuracy_of_accepted_known']):>18}")
    log("(known flagged should be near " + pct(1 - args.keep) + " if validation and test behave alike; "
        "'acc. of accepted' = closed-set accuracy on known test images the detector did not flag)")
    (out / "ood_results.json").write_text(json.dumps(
        {"known": known, "unknown": unknown, "keep": args.keep, "splits_csv": splits_csv,
         "n_known_test": n_kn, "n_unknown_test": n_unk, "closed_set_accuracy": float(known_correct.mean()),
         "scores": results}, indent=2))

    # ---- C. breakdown by source folder ----
    log("\n=== C. BREAKDOWN BY SOURCE FOLDER (flag rate; for known classes this is the false-alarm rate) ===")
    rows = []
    for (label, source), g in te.groupby(["label", "source"]):
        idx = g.index.values
        row = {"label": label, "source": source, "role": "unknown" if label in unknown else "known", "n": len(idx)}
        for name, s in scores.items():
            row[f"flagged_{name}"] = float((s[idx] > thr[name]).mean())
        rows.append(row)
    br = pd.DataFrame(rows)
    br.to_csv(out / "by_source.csv", index=False)
    show = br.copy()
    for c in show.columns:
        if c.startswith("flagged_"):
            show[c] = show[c].map(pct)
    log(show.to_string(index=False))
    log("README: if the unknown class is flagged far more often in one source folder than the other, detection\n"
        "depends on photo conditions, not just on the disease. Likewise for false alarms on known classes.")

    # ---- per-image scores (useful later for calibration / few-shot work) ----
    per = te[["rel_path", "label", "source"]].copy()
    per["is_unknown"] = is_unk
    per["pred_class"] = [known[i] for i in pred]
    per["confidence"] = conf
    for name, s in scores.items():
        per[name] = s
    per.to_csv(out / "per_image_scores.csv", index=False)

    # ---- plots ----
    names = list(scores)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        s = scores[name]
        ax.hist(s[known_mask], bins=50, alpha=0.55, density=True, label="known (test)")
        ax.hist(s[is_unk], bins=50, alpha=0.55, density=True, label="unknown (test)")
        ax.axvline(thr[name], color="k", ls="--", label=f"threshold ({pct(args.keep)} of known val)")
        ax.set_title(f"{name}  (AUROC {results[name]['auroc']:.3f})")
        ax.set_xlabel("score (higher = more unfamiliar)")
    axes[0][0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "score_histograms.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(is_unk, s)
        ax.plot(fpr, tpr, label=f"{name} ({results[name]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false alarm rate (known flagged)")
    ax.set_ylabel("unknown flagged")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=110)
    plt.close(fig)

    (out / "summary.txt").write_text("\n".join(lines))
    log(f"\nAll outputs are in {out}")


if __name__ == "__main__":
    main()
