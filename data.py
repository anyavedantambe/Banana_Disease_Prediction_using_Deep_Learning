"""Data utilities: scan folders, make a FIXED train/val/test split, build tf.data pipelines.

Expected layout (unzip and put the class folders under one root):

    data/banana/
        HEALTHY-1/        -> class "healthy"
        HEALTHY-2/
        BLACK SIGATOKA-1/ -> class "black_sigatoka"
        FUSARIUM WILT-1/  -> class "fusarium_wilt"
        ...

Folder names are normalized: trailing numbers are stripped, spaces become underscores,
everything is lowercased. Images may be nested anywhere inside each folder.
"""
import re
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png"}
CROP_SCALE = 1.15  # training images are loaded 15% larger, then randomly cropped
AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------------
# Scanning and splitting
# ----------------------------------------------------------------------------
def normalize_class_name(folder_name: str) -> str:
    name = re.sub(r"[\s_-]*\d+$", "", folder_name.strip())  # "FUSARIUM WILT-2" -> "FUSARIUM WILT"
    return re.sub(r"[\s-]+", "_", name).lower()


def scan_dataset(root: str) -> pd.DataFrame:
    """Return a DataFrame with columns: rel_path, label, source, path.

    rel_path is relative to `root` (forward slashes) and is what gets stored in splits.csv, so the
    split file works on any machine. `path` is the full path on THIS machine. Rows are sorted by
    rel_path so the same seed gives the same split everywhere.
    """
    root_p = Path(root)
    rows = []
    for class_dir in sorted(p for p in root_p.iterdir() if p.is_dir()):
        label = normalize_class_name(class_dir.name)
        for p in class_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({"rel_path": p.relative_to(root_p).as_posix(),
                             "label": label, "source": class_dir.name})
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")
    df = pd.DataFrame(rows).sort_values("rel_path").reset_index(drop=True)
    df["path"] = [str(root_p / r) for r in df["rel_path"]]
    return df


def make_splits(df, root, val_frac=0.15, test_frac=0.15, seed=42, group_col=None, split_csv=None):
    """Create (or load) a fixed split. The test set is made ONCE and reused by every experiment.

    split_csv stores only relative paths, so commit it to git and reuse it on every machine
    (laptop, Colab, AWS). If it exists it is loaded as-is; delete it only if the dataset changed.

    group_col: optional column (e.g. farm/plot/session id) so that images from the same group
    never straddle train/val/test. The default (None) is a stratified random split, which can
    leak near-duplicate images across splits -- add metadata and use group_col if you can.
    """
    root_p = Path(root)
    if split_csv and Path(split_csv).exists():
        full = pd.read_csv(split_csv)
        full["path"] = [str(root_p / r) for r in full["rel_path"]]
        missing = [p for p in full["path"] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files listed in {split_csv} were not found under {root} "
                f"(e.g. {missing[0]}). Wrong --data_root, or the dataset changed? "
                f"If the dataset changed, delete {split_csv} to create a new split.")
        extra = len(df) - len(full)
        if extra > 0:
            print(f"Note: {extra} images under {root} are not in {split_csv} and will be ignored.")
        return tuple(full[full["split"] == s].drop(columns="split").reset_index(drop=True)
                     for s in ("train", "val", "test"))

    if group_col is None:
        train_val, test = train_test_split(
            df, test_size=test_frac, stratify=df["label"], random_state=seed)
        train, val = train_test_split(
            train_val, test_size=val_frac / (1 - test_frac),
            stratify=train_val["label"], random_state=seed)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        tv_idx, test_idx = next(gss.split(df, groups=df[group_col]))
        train_val, test = df.iloc[tv_idx], df.iloc[test_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
        tr_idx, va_idx = next(gss2.split(train_val, groups=train_val[group_col]))
        train, val = train_val.iloc[tr_idx], train_val.iloc[va_idx]

    if split_csv:
        cols = ["rel_path", "label", "source"]
        out = pd.concat([train[cols].assign(split="train"),
                         val[cols].assign(split="val"),
                         test[cols].assign(split="test")])
        Path(split_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(split_csv, index=False)
        print(f"Saved new split to {split_csv} (commit this file to git).")
    return tuple(d.reset_index(drop=True) for d in (train, val, test))


def subsample_per_class(df, n, seed=42):
    """Keep at most n images per class (for smoke tests now, few-shot experiments later)."""
    parts = [g.sample(min(len(g), n), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts)


def class_weights(df, class_names):
    """Balanced class weights as a {class_index: weight} dict for model.fit."""
    labels = df[df["label"].isin(class_names)]["label"].map({c: i for i, c in enumerate(class_names)})
    w = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=labels.values)
    return {i: float(x) for i, x in enumerate(w)}


# ----------------------------------------------------------------------------
# tf.data pipeline
# ----------------------------------------------------------------------------
def preprocess(img):
    """VGG16 preprocessing (RGB->BGR, ImageNet mean subtraction). Input: float32 in [0, 255].
    Any inference code later (OOD scoring, new images) must apply this same function."""
    return tf.keras.applications.vgg16.preprocess_input(img)


def make_dataset(df, class_names, img_size=224, batch_size=32, training=False,
                 cache=False, seed=42):
    """Build a batched dataset of (preprocessed image, integer label).

    class_names defines the label order AND which classes are included, so the same function
    handles the full 3-class problem and the held-out-class experiments later:
        class_names=["healthy", "black_sigatoka"]  # fusarium_wilt rows are dropped here
    Evaluation datasets are NOT shuffled, so predictions line up with df's row order.
    """
    df = df[df["label"].isin(class_names)]
    idx = {c: i for i, c in enumerate(class_names)}
    paths = df["path"].values
    labels = df["label"].map(idx).values.astype("int32")
    load_size = int(round(img_size * CROP_SCALE)) if training else img_size

    def _load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, [load_size, load_size], antialias=True)  # squashes 4:3 to square
        return tf.saturate_cast(img, tf.uint8), label

    def _augment(img, label):
        img = tf.image.random_crop(img, [img_size, img_size, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.cast(img, tf.float32)
        img = tf.image.random_brightness(img, 0.15 * 255)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        return tf.clip_by_value(img, 0.0, 255.0), label

    def _to_float(img, label):
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training and not cache:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)  # shuffle paths: cheap
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()  # caches resized uint8 images in RAM (~1.7 GB for 11k images at 224px)
        if training:
            ds = ds.shuffle(1000, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_augment if training else _to_float, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)
