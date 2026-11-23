"""Rebuild a grouped (blocks) split around a KNOWN test set.

Use this when a regenerated split doesn't match the one behind your reported results (e.g. a
different scikit-learn version picked different blocks). The old TEST images are kept exactly;
train/val are rebuilt from everything else, grouped by (source folder, file-number block).

    python rebuild_split.py --data_root data/banana_small --old_test_csv nn_pairs.csv

--old_test_csv: any CSV with a column of relative paths of the old test images
(nn_pairs.csv has 'test_file'; a split CSV has 'rel_path'). Output: splits/splits_blocks100.csv
(refuses to overwrite an existing file unless --force).

Safety checks: all old test files must exist under --data_root; groups that would straddle
test and non-test are quarantined (dropped from train/val) so nothing leaks.
"""
import argparse
from pathlib import Path

import pandas as pd

import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--old_test_csv", required=True)
    ap.add_argument("--test_col", default=None, help="column with old test paths (auto: test_file or rel_path)")
    ap.add_argument("--out", default="splits/splits_blocks100.csv")
    ap.add_argument("--block_size", type=int, default=100)
    ap.add_argument("--val_share_of_rest", type=float, default=0.15 / 0.85,
                    help="share of the non-test images used for validation (default matches make_splits)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    if Path(args.out).exists() and not args.force:
        raise SystemExit(f"{args.out} already exists. Delete it or pass --force to overwrite.")

    old = pd.read_csv(args.old_test_csv)
    col = args.test_col or next((c for c in ("test_file", "rel_path") if c in old.columns), None)
    if col is None:
        raise SystemExit(f"No test-path column found in {args.old_test_csv}; columns: {list(old.columns)}")
    old_test = set(old[col])
    print(f"Old test list: {len(old_test)} images (column '{col}')")

    df = data.add_groups(data.scan_dataset(args.data_root), args.block_size)
    print(f"Found {len(df)} images under {args.data_root}")

    missing = old_test - set(df["rel_path"])
    if missing:
        raise SystemExit(f"{len(missing)} old test files were not found under {args.data_root} "
                         f"(e.g. {sorted(missing)[0]}). Is --data_root the same dataset/folder layout "
                         f"as the original run?")

    test = df[df["rel_path"].isin(old_test)]
    rest = df[~df["rel_path"].isin(old_test)]

    straddle = set(test["group"]) & set(rest["group"])
    if straddle:
        n_drop = int(rest["group"].isin(straddle).sum())
        print(f"WARNING: {len(straddle)} blocks contain both old-test and other images "
              f"(the dataset differs from the original?). Dropping {n_drop} non-test images from "
              f"those blocks so nothing leaks into training.")
        rest = rest[~rest["group"].isin(straddle)]

    train, val = data._stratified_group_split(rest, "group", args.val_share_of_rest, args.seed)

    cols = ["rel_path", "label", "source"]
    out = pd.concat([train[cols].assign(split="train"), val[cols].assign(split="val"),
                     test[cols].assign(split="test")])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # ---- checks ----
    groups = {n: set(d["group"]) for n, d in (("train", train), ("val", val), ("test", test))}
    assert not (groups["train"] & groups["test"]) and not (groups["val"] & groups["test"]) \
        and not (groups["train"] & groups["val"]), "a block ended up in two splits"
    assert set(test["rel_path"]) == old_test, "test set differs from the old one"
    print(f"\nSaved {args.out}: train={len(train)} val={len(val)} test={len(test)} (test set identical to the old one)")
    print(pd.crosstab(out["label"], out["split"])[["train", "val", "test"]].to_string())
    print("\nNext: commit this file to git, and re-run the baseline once on it so your reported numbers "
          "correspond to a split anyone can reproduce.")


if __name__ == "__main__":
    main()
