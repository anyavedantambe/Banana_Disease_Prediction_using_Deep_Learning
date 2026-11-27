"""Look at the images behind a cross-dataset result: what do the model's mistakes look like?

    python cross_dataset_examples.py --run_dir /content/drive/MyDrive/banana_results/vgg16_blocks100 \
        --manifest banana_leaf --data_root /content/data/banana_leaf_small

Reads <run_dir>/cross_dataset/<manifest>_predictions.csv (written by cross_dataset_eval.py) and saves,
for every class of the other dataset, one grid of WRONG predictions (most confident first) and one grid
of RANDOM CORRECT predictions to <run_dir>/cross_dataset/examples/. Captions give the model's guess and
confidence. Needs only pandas, Pillow and matplotlib (no TensorFlow).

What to look for: do the wrong images look like a different disease than their label says (label
mismatch, e.g. yellow vs black Sigatoka), or like the right disease photographed differently
(domain shift: background, lighting, distance, leaf age)?
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def grid(items, path, title, ncols=6):
    if not items:
        return
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (p, cap) in zip(axes.ravel(), items):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(cap, fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--manifest", required=True, help="manifest name without .csv (e.g. banana_leaf)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--n", type=int, default=12, help="images per grid")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(args.run_dir) / "cross_dataset"
    pred = pd.read_csv(base / f"{args.manifest}_predictions.csv")
    out = base / "examples"
    out.mkdir(exist_ok=True)
    root = Path(args.data_root)

    for native, g in pred.groupby("native"):
        wrong = g[g["pred"] != g["label"]].sort_values("confidence", ascending=False).head(args.n)
        right = g[g["pred"] == g["label"]]
        right = right.sample(min(args.n, len(right)), random_state=args.seed) if len(right) else right
        safe = "".join(c if c.isalnum() else "_" for c in native)

        def items(df):
            return [(root / r.rel_path, f"-> {r.pred} ({r.confidence:.2f})\n{r.source[:28]}")
                    for r in df.itertuples()]

        grid(items(wrong), out / f"{safe}_WRONG.png",
             f"{native}: WRONG ({int((g['pred'] != g['label']).sum())} of {len(g)}); expected '{g['label'].iloc[0]}', most confident first")
        grid(items(right), out / f"{safe}_RIGHT.png",
             f"{native}: random CORRECT predictions (expected '{g['label'].iloc[0]}')")
        print(f"{native}: {len(g)} images, {int((g['pred'] != g['label']).sum())} wrong -> saved grids")
    print(f"\nGrids are in {out}")


if __name__ == "__main__":
    main()
