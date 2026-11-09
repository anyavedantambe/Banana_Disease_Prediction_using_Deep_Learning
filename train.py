"""Two-phase transfer learning (train head, then fine-tune top of VGG16) with resume support.

Example (PyCharm: Run > Edit Configurations > Parameters):
    --data_root data/banana_small --run_name vgg16_3class
Quick pipeline check:
    --data_root data/banana_small --run_name smoke --max_per_class 60 --epochs1 1 --epochs2 1
Closed-set run for a holdout experiment later (fusarium_wilt rows are excluded from training):
    --data_root data/banana_small --run_name holdout_fusarium --classes healthy,black_sigatoka
On Colab, keep results on Drive so a disconnect loses nothing:
    --out_root /content/drive/MyDrive/banana_results

RESUMING: if the process dies (Colab disconnect, closed laptop), run the SAME command again and
it continues from the last finished epoch. Use --fresh to discard a run and start over. Don't
change hyperparameters between a stop and a resume; use a new --run_name instead.
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

import data
import models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--run_name", default="vgg16_run")
    p.add_argument("--out_root", default="results", help="where run folders are written")
    p.add_argument("--split_mode", default="random", choices=["random", "blocks", "folder"],
                   help="random = by image; blocks = keep (folder, file-number block) groups together; "
                        "folder = hold out whole _<holdout_suffix> folders as the test set")
    p.add_argument("--block_size", type=int, default=100, help="file-number block width for grouped splits")
    p.add_argument("--holdout_suffix", default="2", help="folder mode: folders ending _<this> become the test set")
    p.add_argument("--splits_csv", default=None,
                   help="fixed split file; default depends on --split_mode. Commit it to git.")
    p.add_argument("--classes", default=None, help="comma-separated; default = all classes found")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs1", type=int, default=10, help="phase 1: head only")
    p.add_argument("--epochs2", type=int, default=15, help="phase 2: fine-tune")
    p.add_argument("--lr1", type=float, default=1e-3)
    p.add_argument("--lr2", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--head_units", type=int, default=256)
    p.add_argument("--unfreeze", default="block5", help="comma-separated, e.g. block5,block4")
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (for offline tests)")
    p.add_argument("--max_per_class", type=int, default=None, help="subsample for smoke tests")
    p.add_argument("--cache", action="store_true", help="cache resized images in RAM")
    p.add_argument("--fresh", action="store_true", help="delete any saved progress for this run")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------------------------------------------------------
# Resume machinery
# ----------------------------------------------------------------------------
class RunState:
    """Progress saved to out_dir/state.json after every epoch."""

    def __init__(self, out_dir):
        self.path = out_dir / "state.json"
        self.d = {"phase1_done": False, "phase2_done": False, "epochs_done": 0,
                  "history": {"phase1": [], "phase2": []}}
        if self.path.exists():
            self.d = json.loads(self.path.read_text())

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, default=float))
        os.replace(tmp, self.path)  # atomic: a crash can't leave half a file


class EpochCheckpoint(keras.callbacks.Callback):
    """After every epoch: save the full model (weights + optimizer) and the progress file."""

    def __init__(self, out_dir, state, phase):
        super().__init__()
        self.out_dir, self.state, self.phase = out_dir, state, phase

    def on_epoch_end(self, epoch, logs=None):
        tmp = self.out_dir / "last.tmp.keras"
        self.model.save(tmp)
        os.replace(tmp, self.out_dir / "last.keras")  # model first, then progress
        self.state.d["epochs_done"] = epoch + 1
        self.state.d["history"][f"phase{self.phase}"].append(
            {k: float(v) for k, v in (logs or {}).items()})
        self.state.save()


def callbacks(out_dir, state, phase, patience):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(1, patience - 1)),
        EpochCheckpoint(out_dir, state, phase),  # last, so it sees the final logs
    ]


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_root) / args.run_name
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(out_dir)
    if state.d["phase1_done"] or state.d["epochs_done"]:
        print(f"Resuming run '{args.run_name}': phase1_done={state.d['phase1_done']}, "
              f"epochs_done_in_current_phase={state.d['epochs_done']}")

    # ---- data: the split is fixed and shared by ALL experiments ----
    df = data.scan_dataset(args.data_root)
    if args.splits_csv is None:
        args.splits_csv = {"random": "splits/splits.csv",
                           "blocks": f"splits/splits_blocks{args.block_size}.csv",
                           "folder": f"splits/splits_folder{args.holdout_suffix}.csv"}[args.split_mode]
    print("Using split file:", args.splits_csv)
    train_df, val_df, test_df = data.make_splits(
        df, args.data_root, split_csv=args.splits_csv, mode=args.split_mode,
        block_size=args.block_size, holdout_suffix=args.holdout_suffix)
    if args.max_per_class:
        train_df, val_df, test_df = (data.subsample_per_class(d, args.max_per_class, args.seed)
                                     for d in (train_df, val_df, test_df))
    class_names = args.classes.split(",") if args.classes else sorted(df["label"].unique())
    print("Classes:", class_names)
    print(df.groupby("label").size().to_string())

    kw = dict(img_size=args.img_size, batch_size=args.batch_size, cache=args.cache, seed=args.seed)
    train_ds = data.make_dataset(train_df, class_names, training=True, **kw)
    val_ds = data.make_dataset(val_df, class_names, training=False, **kw)
    test_ds = data.make_dataset(test_df, class_names, training=False, **kw)
    cw = data.class_weights(train_df, class_names)
    unfreeze = tuple(args.unfreeze.split(","))

    # ---- PHASE 1: backbone frozen, train only the new head ----
    if not state.d["phase1_done"]:
        if state.d["epochs_done"] > 0 and (out_dir / "last.keras").exists():
            model = keras.models.load_model(out_dir / "last.keras")  # includes optimizer state
        else:
            state.d["epochs_done"] = 0
            state.d["history"]["phase1"] = []
            model, _ = models.build_vgg16(len(class_names), args.img_size, args.head_units,
                                          args.dropout, None if args.weights == "none" else args.weights)
            models.compile_model(model, args.lr1)
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs1,
                  initial_epoch=state.d["epochs_done"], class_weight=cw, shuffle=False, verbose=2,
                  callbacks=callbacks(out_dir, state, 1, patience=3))
        model.save(out_dir / "phase1_final.keras")  # holds EarlyStopping's restored best weights
        state.d["phase1_done"], state.d["epochs_done"] = True, 0
        state.save()

    # ---- PHASE 2: unfreeze top block(s), recompile with a MUCH smaller learning rate ----
    if not state.d["phase2_done"]:
        if state.d["epochs_done"] > 0 and (out_dir / "last.keras").exists():
            model = keras.models.load_model(out_dir / "last.keras")  # mid-phase-2 resume
            models.unfreeze_blocks(models.get_base(model), unfreeze)
        else:
            model = keras.models.load_model(out_dir / "phase1_final.keras")
            models.unfreeze_blocks(models.get_base(model), unfreeze)
            models.compile_model(model, args.lr2)  # recompiling is required after changing trainable flags
            state.d["history"]["phase2"] = []
            state.d["epochs_done"] = 0
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs2,
                  initial_epoch=state.d["epochs_done"], class_weight=cw, shuffle=False, verbose=2,
                  callbacks=callbacks(out_dir, state, 2, patience=4))
        model.save(out_dir / "model.keras")
        state.d["phase2_done"] = True
        state.save()
    else:
        model = keras.models.load_model(out_dir / "model.keras")

    # ---- evaluate on the fixed test set (closed-set classes only) ----
    logits = model.predict(test_ds, verbose=0)
    y_true = (test_df[test_df["label"].isin(class_names)]["label"]
              .map({c: i for i, c in enumerate(class_names)}).values)
    y_pred = logits.argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(report)
    print(cm)

    # ---- save everything later experiments (OOD, calibration) will need ----
    np.save(out_dir / "test_logits.npy", logits)
    (out_dir / "report.txt").write_text(report + "\n" + np.array2string(cm))
    hist = {ph: {k: [e[k] for e in epochs] for k in epochs[0]} if epochs else {}
            for ph, epochs in state.d["history"].items()}
    (out_dir / "history.json").write_text(json.dumps({**hist, "args": vars(args)}, indent=2))
    for f in ("last.keras", "last.tmp.keras"):  # no longer needed once the run is complete
        (out_dir / f).unlink(missing_ok=True)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
