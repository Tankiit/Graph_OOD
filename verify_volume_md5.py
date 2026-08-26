#!/usr/bin/env python3
"""
verify_volume_md5.py -- ground-truth checksums, computed where the data lives.

`modal volume get` fetches large files with parallel range requests and has been
observed returning a full-size file with zeroed holes while printing success.
`torch.load` does not verify CRCs, so a holed file loads and yields quietly wrong
numbers. The only way to know a local copy is good is to compare it against a
checksum computed server-side.

    modal run verify_volume_md5.py --paths llama3_8b/truthfulqa/contrastive_h.pt
"""

import hashlib

import modal

app = modal.App("ova-arr-verify")
vol = modal.Volume.from_name("ova-arr-extract-full")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def checksums(paths: list[str]) -> dict:
    out = {}
    for rel in paths:
        p = f"/data/{rel}"
        h = hashlib.md5()
        n = 0
        zeros = 0
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                h.update(b)
                n += len(b)
                zeros += b.count(0)
        out[rel] = dict(md5=h.hexdigest(), size=n, zero_frac=zeros / max(n, 1))
    return out


@app.local_entrypoint()
def main(paths: str = "llama3_8b/truthfulqa/contrastive_h.pt"):
    for rel, info in checksums.remote(paths.split(",")).items():
        print(f"{info['md5']}  size={info['size']}  "
              f"zero_frac={info['zero_frac']:.4%}  {rel}")
