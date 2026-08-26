#!/usr/bin/env python3
"""
fetch_volume_verified.py -- download from a Modal volume without silent corruption.

WHY THIS EXISTS
    `modal volume get` (client 1.3.0) fetches large files with parallel range
    requests. Observed repeatedly on `ova-arr-extract-full`: it prints
    "Finished downloading files to local!", writes a file of exactly the right
    size, and leaves multi-megabyte runs of zeros inside it. Three fetches of
    the same 1.28 GB file gave three different md5s, none matching the server.

    Nothing downstream catches this. `torch.load` does not verify CRCs, so a
    holed tensor loads fine and quietly changes your numbers (pair accuracy
    moved 0.933 -> 0.927 between two copies of the same file). And
    `zipfile.testzip()` is useless on `.pt` files because PyTorch writes zero
    CRCs, so it reports every healthy file as broken.

    Small files (~200 KB) came through intact every time. So: split server-side
    into parts, fetch each part, verify each part against a server-computed
    md5, retry the ones that fail, then concatenate and verify the whole.

USAGE
    modal run fetch_volume_verified.py --paths llama3_8b/truthfulqa/contrastive_h.pt \
        --dest /tmp/ova_arr_full/full
"""

import hashlib
import os
import shutil
import subprocess

import modal

VOLUME = "ova-arr-extract-full"
CHUNK_DIR = "_verified_chunks"

app = modal.App("ova-arr-fetch-verified")
vol = modal.Volume.from_name(VOLUME)
image = modal.Image.debian_slim(python_version="3.11")


def _md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def split_remote(rel: str, part_bytes: int) -> dict:
    """Write `rel` as fixed-size parts inside the volume; return their md5s."""
    src = f"/data/{rel}"
    out_dir = f"/data/{CHUNK_DIR}/{rel.replace('/', '__')}"
    os.makedirs(out_dir, exist_ok=True)

    parts, whole = [], hashlib.md5()
    with open(src, "rb") as f:
        i = 0
        while True:
            b = f.read(part_bytes)
            if not b:
                break
            whole.update(b)
            name = f"part_{i:05d}"
            with open(f"{out_dir}/{name}", "wb") as g:
                g.write(b)
            parts.append(dict(name=name, md5=hashlib.md5(b).hexdigest(),
                              size=len(b)))
            i += 1
    vol.commit()
    return dict(dir=f"{CHUNK_DIR}/{rel.replace('/', '__')}", parts=parts,
                md5=whole.hexdigest(), size=os.path.getsize(src))


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def cleanup_remote(rel: str):
    d = f"/data/{CHUNK_DIR}/{rel.replace('/', '__')}"
    shutil.rmtree(d, ignore_errors=True)
    vol.commit()


def _get(remote_path, local_dir):
    subprocess.run(["modal", "volume", "get", "--force", VOLUME,
                    remote_path, local_dir + "/"],
                   check=True, capture_output=True)


@app.local_entrypoint()
def main(paths: str = "llama3_8b/truthfulqa/contrastive_h.pt",
         dest: str = "/tmp/ova_arr_full/full",
         part_mb: int = 64,
         retries: int = 6):
    for rel in paths.split(","):
        rel = rel.strip()
        print(f"\n=== {rel} ===")
        info = split_remote.remote(rel, part_mb * 1024 * 1024)
        print(f"  server md5 {info['md5']}  {info['size']} bytes  "
              f"{len(info['parts'])} parts")

        out_path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        staging = os.path.join(dest, ".parts", rel.replace("/", "__"))
        os.makedirs(staging, exist_ok=True)

        try:
            for p in info["parts"]:
                local = os.path.join(staging, p["name"])
                for attempt in range(1, retries + 1):
                    if os.path.exists(local) and _md5(local) == p["md5"]:
                        break
                    if os.path.exists(local):
                        os.remove(local)
                    _get(f"{info['dir']}/{p['name']}", staging)
                    if _md5(local) == p["md5"]:
                        if attempt > 1:
                            print(f"  {p['name']}: ok after {attempt} attempts")
                        break
                    print(f"  {p['name']}: md5 mismatch, retry {attempt}")
                else:
                    raise RuntimeError(f"{rel}:{p['name']} failed after "
                                       f"{retries} attempts")

            with open(out_path, "wb") as out:
                for p in info["parts"]:
                    with open(os.path.join(staging, p["name"]), "rb") as f:
                        shutil.copyfileobj(f, out, 1 << 22)

            got = _md5(out_path)
            ok = got == info["md5"]
            print(f"  {'VERIFIED' if ok else 'FAILED'}  {got}  -> {out_path}")
            if not ok:
                raise RuntimeError(f"{rel}: assembled md5 {got} != {info['md5']}")
            shutil.rmtree(staging, ignore_errors=True)
        finally:
            cleanup_remote.remote(rel)
