"""Download a random subset of OpenImages train images, shrunk on the fly.

    uv run python scripts/download_openimages.py                    # 1M images, short side 256
    uv run python scripts/download_openimages.py -n 1000 --workers 16   # smoke test

Image ids come from the boxable train list (~1.74M images); `n` of them are drawn with a fixed
seed and written to {out}/ids_{n}.txt, so the subset is reproducible. Images are fetched from the
public CVDF S3 bucket, EXIF-rotated, downscaled so the short side is at most `--size`, and saved as
JPEG under {out}/train/{id[:2]}/{id}.jpg. Existing files are skipped, so the script can be
re-run to resume; ids that fail are appended to {out}/failed.txt.
"""

import argparse
import csv
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from actdist.data import DATA_DIR

IDS_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"


def fetch_ids(out: Path, n: int, seed: int) -> list[str]:
    ids_path = out / f"ids_{n}.txt"
    if ids_path.exists():
        return ids_path.read_text().split()
    csv_path = out / "train-images-boxable-with-rotation.csv"
    if not csv_path.exists():
        print(f"downloading id list to {csv_path}")
        urllib.request.urlretrieve(IDS_URL, csv_path)
    with open(csv_path, newline="") as f:
        all_ids = sorted(row["ImageID"] for row in csv.DictReader(f))
    ids = sorted(np.random.default_rng(seed).choice(all_ids, n, replace=False).tolist())
    ids_path.write_text("\n".join(ids))
    print(f"sampled {n} of {len(all_ids)} ids -> {ids_path}")
    return ids


def download_one(image_id: str, dest: Path, size: int, retries: int = 3) -> str | None:
    """Returns None on success, an error message otherwise."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(IMG_URL.format(image_id), timeout=60) as r:
                raw = r.read()
            with Image.open(io.BytesIO(raw)) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                scale = size / min(im.size)
                if scale < 1:
                    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.BICUBIC)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmp")
                im.save(tmp, "JPEG", quality=90)
                tmp.rename(dest)  # atomic: a killed run never leaves a truncated .jpg
            return None
        except Exception as e:  # network errors, corrupt images
            err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    return err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=1_000_000)
    p.add_argument("--size", type=int, default=256, help="max short side in pixels")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DATA_DIR / "openimages")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ids = fetch_ids(args.out, args.n, args.seed)
    img_dir = args.out / "train"
    todo = [i for i in ids if not (img_dir / i[:2] / f"{i}.jpg").exists()]
    print(f"{len(ids) - len(todo)} already present, {len(todo)} to download", flush=True)

    t0, done, failed = time.time(), 0, 0
    fetch = lambda i: download_one(i, img_dir / i[:2] / f"{i}.jpg", args.size)
    with ThreadPoolExecutor(args.workers) as pool, open(args.out / "failed.txt", "a") as flog:
        for start in range(0, len(todo), 5000):  # chunked so 1M futures never exist at once
            chunk = todo[start:start + 5000]
            for image_id, err in zip(chunk, pool.map(fetch, chunk)):
                if err:
                    failed += 1
                    flog.write(f"{image_id}\t{err}\n")
            flog.flush()
            done += len(chunk)
            rate = done / (time.time() - t0)
            print(f"{done}/{len(todo)}  failed {failed}  {rate:.0f} img/s  "
                  f"eta {(len(todo) - done) / rate / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
