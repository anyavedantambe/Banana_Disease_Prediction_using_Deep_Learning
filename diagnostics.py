"""Is a ~99% test accuracy trustworthy? Checks for split leakage and shortcut learning.

    python diagnostics.py --data_root data/banana_small --run_dir /content/drive/MyDrive/banana_results/vgg16_3class

Needs the finished run's model.keras. Writes everything to <run_dir>/diagnostics/:
  1. source_crosstab.csv + printed summary   which source folders feed which split
  2. misclassified.png/.csv, correct_sample.png   LOOK at these (leaf vs stem? backgrounds? style?)
  3. nn_similarity.png, nn_pairs_top.png, nn_pairs_median.png   near-duplicate train/test twins
  4. shortcut_probe.txt   can simple non-disease image statistics already predict the class?
Skip slow parts with --skip_neighbors or --skip_probe.

How to read the results is explained in the printed summary and the README text at the bottom
of each section. Nothing here proves leakage by itself: the image grids are the real evidence.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tensorflow import keras
from tensorflow.keras import layers

import data


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def image_grid(items, path, ncols=6, title=None):
    """items: list of (image_path, caption). Saves a grid PNG."""
    if not items:
        return
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (p, cap) in zip(axes.ravel(), items):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(cap, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def predict_probs(model, df, class_names, img_size, batch_size):
    ds = data.make_dataset(df, class_names, img_size=img_size, batch_size=batch_size, training=False)
    logits = model.predict(ds, verbose=0)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def embeddings(model, df, class_names, img_size, batch_size):
    """512-d global-average-pooled VGG16 features (before the task-specific head)."""
    gap = next(l for l in model.layers if isinstance(l, layers.GlobalAveragePooling2D))
    emb_model = keras.Model(model.inputs, gap.output)
    ds = data.make_dataset(df, class_names, img_size=img_size, batch_size=batch_size, training=False)
    return emb_model.predict(ds, verbose=0)


def simple_features(path):
    """Cheap image statistics that have nothing to do with disease symptoms."""
    p = Path(path)
    with Image.open(p) as im:
        w, h = im.size
        im = im.convert("RGB")
        im.thumbnail((64, 64))
        a = np.asarray(im, dtype=np.float32) / 255.0
    return [w, h, w / h, p.stat().st_size, *a.mean((0, 1)), *a.std((0, 1))]


# ----------------------------------------------------------------------------
# 1. source folders vs splits
# ----------------------------------------------------------------------------
def check_sources(train_df, val_df, test_df, out, log):
    all_df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val"),
                        test_df.assign(split="test")], ignore_index=True)
    ct = pd.crosstab([all_df["label"], all_df["source"]], all_df["split"])[["train", "val", "test"]]
    ct.to_csv(out / "source_crosstab.csv")
    log("\n=== 1. SOURCE FOLDERS vs SPLITS ===")
    log(ct.to_string())
    n_src = all_df.groupby("label")["source"].nunique()
    log("\nSource folders per class:\n" + n_src.to_string())
    mixed = (ct > 0).sum(axis=1)
    log(f"\n{int((mixed == 3).sum())}/{len(ct)} source folders appear in train, val AND test.")
    log("Example files per source folder:")
    for src, g in all_df.groupby("source"):
        log(f"  {src}: " + ", ".join(g["rel_path"].head(2)))
    if n_src.min() >= 3:
        log("-> Each class has >=3 source folders: a grouped split (hold out whole folders) is feasible.")
    else:
        log("-> Some class has <3 source folders: folder-level grouping isn't possible. If the folders are\n"
            "   not meaningful, look for session/farm/plant IDs in the FILENAMES instead.")
    log("README: a random split puts images from the same folder (often one session/place/plant batch)\n"
        "into both train and test. If folders = sessions, test accuracy can reflect session\n"
        "recognition, not disease recognition.")


# ----------------------------------------------------------------------------
# 2. errors and samples
# ----------------------------------------------------------------------------
def check_errors(test_df, probs, class_names, out, log, seed):
    t = test_df[test_df["label"].isin(class_names)].reset_index(drop=True)
    y_true = t["label"].map({c: i for i, c in enumerate(class_names)}).values
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    t = t.assign(pred=[class_names[i] for i in y_pred], confidence=conf, correct=(y_true == y_pred))

    wrong = t[~t["correct"]].sort_values("confidence", ascending=False)
    wrong[["rel_path", "source", "label", "pred", "confidence"]].to_csv(out / "misclassified.csv", index=False)
    log("\n=== 2. ERRORS AND SAMPLES ===")
    log(f"{len(wrong)} misclassified of {len(t)} test images ({100 * len(wrong) / len(t):.2f}%).")
    log(f"Confidence on errors: median {wrong['confidence'].median():.2f}; "
        f"{int((wrong['confidence'] > 0.9).sum())} errors have confidence > 0.9.")
    image_grid([(r.path, f"true: {r.label}\npred: {r.pred} ({r.confidence:.2f})")
                for r in wrong.head(24).itertuples()], out / "misclassified.png",
               title="Misclassified test images (most confident first)")

    rng = np.random.default_rng(seed)
    items = []
    for cls in class_names:
        g = t[(t["label"] == cls) & t["correct"]]
        for r in g.iloc[rng.permutation(len(g))[:8]].itertuples():
            items.append((r.path, f"{r.label}\n({r.confidence:.2f}) {r.source}"))
    image_grid(items, out / "correct_sample.png", ncols=8,
               title="Random CORRECT test images, 8 per class (caption: class, confidence, source folder)")
    log("Saved misclassified.png and correct_sample.png.")
    log("README: look at correct_sample.png class by class. Do the classes differ in ways unrelated to\n"
        "disease: leaf vs stem, background (soil, sky, hand), lighting, distance, camera style? Are the\n"
        "errors genuinely ambiguous, or do they look easy? Confident errors deserve a close look (label noise?).")
    return t


# ----------------------------------------------------------------------------
# 3. near-duplicates
# ----------------------------------------------------------------------------
def check_neighbors(model, train_df, test_df, class_names, img_size, batch_size, out, log, seed):
    tr = train_df[train_df["label"].isin(class_names)].reset_index(drop=True)
    te = test_df[test_df["label"].isin(class_names)].reset_index(drop=True)
    log("\n=== 3. NEAR-DUPLICATES (nearest training image for each test image) ===")
    f_tr = embeddings(model, tr, class_names, img_size, batch_size)
    f_te = embeddings(model, te, class_names, img_size, batch_size)
    mu = f_tr.mean(0, keepdims=True)  # center first: raw post-ReLU features all look similar
    f_tr, f_te = f_tr - mu, f_te - mu
    f_tr /= np.linalg.norm(f_tr, axis=1, keepdims=True) + 1e-8
    f_te /= np.linalg.norm(f_te, axis=1, keepdims=True) + 1e-8

    best_sim = np.empty(len(te))
    best_idx = np.empty(len(te), dtype=int)
    for i in range(0, len(te), 512):  # chunked so memory stays small
        s = f_te[i:i + 512] @ f_tr.T
        best_idx[i:i + 512] = s.argmax(1)
        best_sim[i:i + 512] = s.max(1)

    same = (tr["label"].values[best_idx] == te["label"].values)
    log(f"Nearest training image has the same class for {100 * same.mean():.1f}% of test images.")
    for thr in (0.80, 0.90, 0.95):
        log(f"  test images with a training neighbor at cosine similarity >= {thr}: "
            f"{100 * (best_sim >= thr).mean():.1f}%")
    diff_hi = int(((best_sim >= 0.9) & ~same).sum())
    log(f"  very similar neighbors with a DIFFERENT class (possible label noise): {diff_hi}")

    # --- do visually similar images have nearby filename numbers? (e.g. Image_216 vs Image_217) ---
    def img_num(rel):
        m = re.findall(r"\d+", Path(rel).stem)
        return float(m[-1]) if m else np.nan

    te_n = te["rel_path"].map(img_num).values
    tr_n = tr["rel_path"].map(img_num).values
    gap = np.abs(te_n - tr_n[best_idx])
    pd.DataFrame({
        "test_file": te["rel_path"], "test_label": te["label"], "test_num": te_n,
        "nn_train_file": tr["rel_path"].values[best_idx], "nn_train_label": tr["label"].values[best_idx],
        "nn_train_num": tr_n[best_idx], "similarity": best_sim, "number_gap": gap,
    }).sort_values("similarity", ascending=False).to_csv(out / "nn_pairs.csv", index=False)
    log("\n--- Filename numbers of nearest-neighbor pairs (full table: nn_pairs.csv) ---")
    if np.isnan(gap).mean() > 0.5:
        log("Could not read a number from most filenames, skipping this check.")
    else:
        rng = np.random.default_rng(seed)
        rand_gap = np.full(len(te), np.nan)
        for cls in class_names:  # baseline: a RANDOM training image of the same class
            ti = np.where(te["label"].values == cls)[0]
            ri = np.where(tr["label"].values == cls)[0]
            rand_gap[ti] = np.abs(te_n[ti] - tr_n[rng.choice(ri, len(ti))])
        log(f"{'filename-number gap <=':<26}{'nearest neighbor':>18}{'random same-class':>20}")
        for k in (5, 20, 100):
            log(f"{k:<26}{100 * np.nanmean(gap <= k):>17.1f}%{100 * np.nanmean(rand_gap <= k):>19.1f}%")
        log(f"{'median gap':<26}{np.nanmedian(gap):>18.0f}{np.nanmedian(rand_gap):>20.0f}")
        log("Ten most similar pairs (test file number -> its nearest train file number):")
        top = np.argsort(-best_sim)[:10]
        for j in top:
            log(f"  sim {best_sim[j]:.3f}  {te['rel_path'].iloc[j]} -> {tr['rel_path'].values[best_idx[j]]}"
                f"  (gap {gap[j]:.0f})")
        log("README: if nearest neighbors are far more often within a few numbers of each other than random\n"
            "same-class images, then neighboring file numbers look alike (probably consecutive shots of the\n"
            "same scene). A random split then puts such neighbors into train AND test: that is leakage.\n"
            "If both columns look similar, file numbers carry no visual meaning and you can drop this worry.")

    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in class_names:
        ax.hist(best_sim[te["label"].values == cls], bins=40, alpha=0.5, label=cls)
    ax.set_xlabel("cosine similarity to nearest training image (features centered)")
    ax.set_ylabel("# test images")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "nn_similarity.png", dpi=110)
    plt.close(fig)

    def pair_items(order):
        items = []
        for j in order:
            a, b = te.iloc[j], tr.iloc[best_idx[j]]
            items.append((a["path"], f"TEST {a['label']}\n{a['source']}"))
            items.append((b["path"], f"TRAIN {b['label']}\nsim {best_sim[j]:.3f}  {b['source']}"))
        return items

    order = np.argsort(-best_sim)
    image_grid(pair_items(order[:8]), out / "nn_pairs_top.png", ncols=4,
               title="Most similar test/train pairs (test left, its nearest train image right)")
    mid = order[len(order) // 2 - 4: len(order) // 2 + 4]
    image_grid(pair_items(mid), out / "nn_pairs_median.png", ncols=4,
               title="Median-similarity pairs, for contrast")
    log("Saved nn_similarity.png, nn_pairs_top.png, nn_pairs_median.png.")
    log("README: open nn_pairs_top.png. If the pairs are the same plant/leaf from slightly different\n"
        "angles, the random split leaks and test accuracy is optimistic. Compare with the median pairs:\n"
        "if even typical pairs look like the same scene, leakage is likely widespread. The thresholds are\n"
        "rough guides only: trust your eyes on the pair images.")


# ----------------------------------------------------------------------------
# 4. shortcut probe
# ----------------------------------------------------------------------------
def check_probe(train_df, test_df, class_names, out, log):
    log("\n=== 4. SHORTCUT PROBE (can non-disease image statistics predict the class?) ===")
    tr = train_df[train_df["label"].isin(class_names)].reset_index(drop=True)
    te = test_df[test_df["label"].isin(class_names)].reset_index(drop=True)
    idx = {c: i for i, c in enumerate(class_names)}
    X_tr = np.array([simple_features(p) for p in tr["path"]])
    X_te = np.array([simple_features(p) for p in te["path"]])
    y_tr, y_te = tr["label"].map(idx).values, te["label"].map(idx).values
    majority = max(np.bincount(y_te)) / len(y_te)
    log(f"Chance level for always guessing the biggest class: {majority:.3f}")
    sets = {"image geometry + file size (width, height, aspect, bytes)": [0, 1, 2, 3],
            "color/brightness statistics only": [4, 5, 6, 7, 8, 9],
            "all of the above": list(range(10))}
    for name, cols in sets.items():
        clf = HistGradientBoostingClassifier(random_state=0).fit(X_tr[:, cols], y_tr)
        pred = clf.predict(X_te[:, cols])
        log(f"\n{name}\n  accuracy {accuracy_score(y_te, pred):.3f}   macro-F1 {f1_score(y_te, pred, average='macro'):.3f}")
        log("  confusion matrix (rows = true, cols = predicted; order: " + ", ".join(class_names) + ")\n"
            + np.array2string(confusion_matrix(y_te, pred), prefix="  "))
    log("\nREADME: a tiny model that never looks at leaf symptoms should score near chance. If it scores far\n"
        "above chance, the classes differ in photo conditions (camera, size, lighting), and a CNN can\n"
        "exploit that too. This doesn't prove YOUR model does, but shows the shortcut is available.\n"
        "(File size is measured on the resized copy, so it is a rough proxy for the original.)")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run_dir", required=True, help="folder containing model.keras")
    ap.add_argument("--splits_csv", default="splits/splits.csv")
    ap.add_argument("--classes", default=None, help="comma-separated, same order as training")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--skip_neighbors", action="store_true")
    ap.add_argument("--skip_probe", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = run_dir / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    df = data.scan_dataset(args.data_root)
    train_df, val_df, test_df = data.make_splits(df, args.data_root, split_csv=args.splits_csv)
    class_names = args.classes.split(",") if args.classes else sorted(df["label"].unique())
    model = keras.models.load_model(run_dir / "model.keras")

    check_sources(train_df, val_df, test_df, out, log)
    probs = predict_probs(model, test_df[test_df["label"].isin(class_names)], class_names,
                          args.img_size, args.batch_size)
    check_errors(test_df, probs, class_names, out, log, args.seed)
    if not args.skip_neighbors:
        check_neighbors(model, train_df, test_df, class_names, args.img_size, args.batch_size, out, log, args.seed)
    if not args.skip_probe:
        check_probe(train_df, test_df, class_names, out, log)

    (out / "summary.txt").write_text("\n".join(lines))
    print(f"\nAll outputs are in {out}")


if __name__ == "__main__":
    main()
