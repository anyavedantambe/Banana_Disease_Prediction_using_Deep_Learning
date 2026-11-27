"""Quick audit of an image dataset laid out as <root>/<class folder>/<images>.

    python dataset_report.py --data_root data/bananalsd

Prints, per class: image count, image sizes, file-name examples and number range. Then finds EXACT
duplicate files (identical bytes), both within a class and across classes (the latter would be a
label conflict). Saves duplicates to <data_root>/../<name>_duplicates.csv if any are found.

Run this on every new dataset before splitting it. It reads files only; it changes nothing.
"""
import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()
    root = Path(args.data_root)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a folder")

    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(class_dir.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except Exception as e:  # unreadable image
                    print(f"UNREADABLE: {p} ({e})")
                    continue
                rows.append({"class": class_dir.name, "path": str(p), "name": p.name, "w": w, "h": h,
                             "md5": md5(p)})
    if not rows:
        raise SystemExit(f"No images found one level below {root}. Is the folder layout <root>/<class>/<images>?")
    df = pd.DataFrame(rows)

    print(f"\n=== {root}: {len(df)} images in {df['class'].nunique()} class folders ===")
    for cls, g in df.groupby("class"):
        sizes = Counter(zip(g["w"], g["h"])).most_common(3)
        nums = g["name"].map(lambda s: (re.findall(r"\d+", Path(s).stem) or [None])[-1])
        nums = pd.to_numeric(nums, errors="coerce").dropna()
        num_txt = f"file numbers {int(nums.min())}..{int(nums.max())}" if len(nums) else "no numbers in file names"
        print(f"\n{cls}: {len(g)} images")
        print("  sizes (w x h): " + ", ".join(f"{w}x{h} (n={n})" for (w, h), n in sizes)
              + (f", +{g[['w', 'h']].drop_duplicates().shape[0] - len(sizes)} other sizes"
                 if g[["w", "h"]].drop_duplicates().shape[0] > len(sizes) else ""))
        print(f"  {num_txt}; examples: " + ", ".join(g["name"].head(3)))

    # ---- exact duplicates ----
    dup = df[df.duplicated("md5", keep=False)].copy()
    print("\n=== EXACT DUPLICATES (identical bytes) ===")
    if dup.empty:
        print("None found.")
    else:
        groups = dup.groupby("md5")
        within = cross = 0
        for _, g in groups:
            if g["class"].nunique() > 1:
                cross += 1
            else:
                within += 1
        redundant = len(dup) - dup["md5"].nunique()
        print(f"{dup['md5'].nunique()} duplicated images; {redundant} redundant extra files to drop.")
        print(f"  within one class: {within} groups | ACROSS classes (label conflict!): {cross} groups")
        per_class = defaultdict(int)
        for _, g in groups:
            for c in g["class"].iloc[1:]:
                per_class[c] += 1
        print("  redundant files per class:", dict(per_class))
        out = root.parent / f"{root.name}_duplicates.csv"
        dup.sort_values(["md5", "path"])[["md5", "class", "path"]].to_csv(out, index=False)
        print(f"  Saved {out}")
    print("\nDone. (Near-duplicates, like the same plant from another angle, need the embedding check in diagnostics.py.)")


if __name__ == "__main__":
    main()
