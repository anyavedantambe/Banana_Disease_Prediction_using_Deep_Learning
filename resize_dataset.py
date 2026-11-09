"""Shrink every image ONCE into a mirrored folder, so training doesn't re-decode huge JPEGs.

    python resize_dataset.py --src data/banana --dst data/banana_small

- The folder layout and file names are preserved exactly, so a splits.csv made on the original
  data also works on the small copy (just change --data_root).
- Aspect ratio is kept: the SHORTER side becomes --short_side pixels (default 288, a bit bigger
  than the 258 px that training crops from). data.py later squashes to square, same as before.
- Safe to re-run: files that already exist in --dst are skipped, so an interrupted run
  (e.g. a Colab disconnect) just continues. Unreadable images are listed in failed.txt.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def resize_one(job):
    src, dst, short_side, quality = job
    try:
        if dst.exists():
            return "skipped", str(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img.draft("RGB", (short_side, short_side))  # fast partial JPEG decode (no-op for PNG)
            img = ImageOps.exif_transpose(img)  # respect phone-camera rotation flags
            img = img.convert("RGB")
            w, h = img.size
            scale = short_side / min(w, h)
            if scale < 1:  # never upscale
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)  # write then rename: no half-files
            if dst.suffix.lower() in {".jpg", ".jpeg"}:
                img.save(tmp, format="JPEG", quality=quality)
            else:
                img.save(tmp, format="PNG")
            tmp.replace(dst)
        return "done", str(src)
    except Exception as e:  # noqa: BLE001 - we want to keep going and report
        return "failed", f"{src}\t{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--short_side", type=int, default=288)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images found under {src}")
    jobs = [(p, dst / p.relative_to(src), args.short_side, args.quality) for p in files]
    print(f"Resizing {len(jobs)} images: {src} -> {dst}")

    counts = {"done": 0, "skipped": 0, "failed": 0}
    failed = []
    with Pool(args.workers) as pool:
        for i, (status, info) in enumerate(pool.imap_unordered(resize_one, jobs, chunksize=16), 1):
            counts[status] += 1
            if status == "failed":
                failed.append(info)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {counts}")

    dst.mkdir(parents=True, exist_ok=True)
    if failed:
        (dst / "failed.txt").write_text("\n".join(failed))
        print(f"{len(failed)} images failed; see {dst / 'failed.txt'}")
    print("Finished:", counts)


if __name__ == "__main__":  # this guard is required for multiprocessing on Windows
    main()
