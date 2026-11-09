"""Quick look at your data pipeline before any training:  python sanity_check.py --data_root data/banana_small

Prints environment info, class counts and split sizes, then saves sanity_batch.png: a grid of
training images AFTER augmentation (colors are un-preprocessed so they look normal).
Check that: labels match what you see, leaves are visible and not cropped away, nothing is odd.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import data

p = argparse.ArgumentParser()
p.add_argument("--data_root", required=True)
p.add_argument("--splits_csv", default="splits/splits.csv")
p.add_argument("--img_size", type=int, default=224)
args = p.parse_args()

print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU") or "none (CPU only)")
df = data.scan_dataset(args.data_root)
train_df, val_df, test_df = data.make_splits(df, args.data_root, split_csv=args.splits_csv)
print("\nImages per class:\n" + df.groupby("label").size().to_string())
print(f"\nSplit sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
print("Test set per class:\n" + test_df.groupby("label").size().to_string())

class_names = sorted(df["label"].unique())
ds = data.make_dataset(train_df, class_names, img_size=args.img_size, batch_size=12, training=True)
x, y = next(iter(ds))
print(f"\nBatch: images {x.shape} {x.dtype}, labels {y.numpy().tolist()}")

mean_bgr = np.array([103.939, 116.779, 123.68])  # undo VGG16 preprocessing for display only
imgs = np.clip((x.numpy() + mean_bgr)[..., ::-1], 0, 255).astype("uint8")
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for ax, im, lab in zip(axes.ravel(), imgs, y.numpy()):
    ax.imshow(im); ax.set_title(class_names[lab], fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig("sanity_batch.png", dpi=100)
print("Saved sanity_batch.png")
