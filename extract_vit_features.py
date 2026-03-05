"""
Feature Extraction: CLIP ViT-G/14 and DINOv2 ViT-G/14
=======================================================
Configured for:
    data_root : /home/tanmoy/research/data/
    GPU       : NVIDIA RTX A4500 (20 GB, Ampere)

Expected dataset layout under data_root:
    CIFAR-10  (or cifar10 / cifar-10-batches-py)
    CIFAR-100 (or cifar100 / cifar-100-python)
    SVHN/                    <- contains test_32x32.mat, train_32x32.mat
    Imagenet_resize/         <- pre-resized ImageNet (OOD set, flat or ImageFolder)
    LSUN_resize/             <- pre-resized LSUN (flat or ImageFolder)
    DTD/                     <- DTD textures  (dtd/images/<cat>/<img>.jpg)
    places365/               <- Places365 extracted (must unzip places365.zip first)
    tiny-imagenet-200/       <- train/ and val/ subdirs

Output layout (all .npz files have keys: features [N,D], labels [N]):
    feature_cache/
        clip_vitg14/
            cifar10_train.npz  cifar10_test.npz
            cifar100_train.npz cifar100_test.npz
            svhn_test.npz
            imagenet_resize.npz
            lsun_resize.npz
            dtd_test.npz
            places365_val.npz
            tinyimagenet_val.npz
        dinov2_vitg14/
            (same)

Usage:
    # Recommended — bf16 on Ampere, resume-safe
    python extract_vit_features.py --use_bf16 --resume

    # One model / subset of datasets
    python extract_vit_features.py --models clip --datasets cifar10 svhn dtd --use_bf16

    # Override paths
    python extract_vit_features.py \
        --data_root /home/tanmoy/research/data \
        --cache_dir /home/tanmoy/research/data/feature_cache \
        --use_bf16 --resume

Requirements:
    pip install open_clip_torch timm tqdm scipy
    (torch + torchvision assumed already installed)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_ROOT  = "/home/tanmoy/research/data"
CACHE_DIR  = "/home/tanmoy/research/data/feature_cache"
BATCH_SIZE = 256    # safe in bf16 on A4500; drop to 128 for fp32
NUM_WORKERS = 8
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# Model loaders
# =============================================================================

def load_clip_vitg14(device):
    """OpenCLIP ViT-G/14 (LAION-2B). Output dim: 1024."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k"
    )
    model = model.visual          # vision encoder only — no text tower
    model.eval().to(device)

    class _Wrapper(nn.Module):
        def __init__(self, v): super().__init__(); self.v = v
        def forward(self, x): return self.v(x)   # [B, 1024]

    return _Wrapper(model), preprocess


def load_dinov2_vitg14(device):
    """DINOv2 ViT-G/14 (torch.hub). Output dim: 1536."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", trust_repo=True)
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    class _Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)   # [B, 1536]

    return _Wrapper(model), preprocess


# =============================================================================
# Path helpers
# =============================================================================

def find_dir(data_root, *candidates):
    """Return first existing path among candidates under data_root."""
    for c in candidates:
        p = Path(data_root) / c
        if p.exists():
            return p
    contents = sorted(x.name for x in Path(data_root).iterdir())
    raise FileNotFoundError(
        f"None of {candidates} found under {data_root}\n"
        f"Actual contents: {contents}"
    )


# =============================================================================
# Dataset builders
# Each returns a torch.utils.data.Dataset
# =============================================================================

def ds_cifar10(split, transform, data_root):
    # Torchvision expects root to be the parent directory of cifar-10-batches-py
    # so we just use data_root if cifar-10-batches-py exists under it
    root = Path(data_root)
    if (root / "cifar-10-batches-py").exists():
        root = root  # use data_root directly
    elif (root / "CIFAR-10" / "cifar-10-batches-py").exists():
        root = root / "CIFAR-10"
    else:
        root = find_dir(data_root, "CIFAR-10", "cifar10")
    return datasets.CIFAR10(str(root), train=(split == "train"),
                            download=False, transform=transform)


def ds_cifar100(split, transform, data_root):
    root = Path(data_root)
    if (root / "cifar-100-python").exists():
        root = root  # use data_root directly
    elif (root / "CIFAR-100" / "cifar-100-python").exists():
        root = root / "CIFAR-100"
    else:
        root = find_dir(data_root, "CIFAR-100", "cifar100")
    return datasets.CIFAR100(str(root), train=(split == "train"),
                             download=False, transform=transform)


def ds_svhn(split, transform, data_root):
    """
    Torchvision SVHN expects <root>/<split>_32x32.mat.
    Your layout: /home/tanmoy/research/data/SVHN/test_32x32.mat  -> root = SVHN/
    """
    root = find_dir(data_root, "SVHN", "svhn")
    return datasets.SVHN(str(root), split=split,
                         download=False, transform=transform)


def ds_imagenet_resize(transform, data_root):
    """
    Imagenet_resize — pre-resized ImageNet images used as OOD reference.
    Can be either:
      (A) flat folder:  Imagenet_resize/<img>.JPEG
      (B) class folders: Imagenet_resize/<class>/<img>.JPEG
    """
    root = find_dir(data_root, "Imagenet_resize", "imagenet_resize",
                    "ImageNet_resize", "Imagenet_Resize")
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        print(f"  [Imagenet_resize] ImageFolder ({len(subdirs)} subdirs)")
        return datasets.ImageFolder(str(root), transform=transform)
    print(f"  [Imagenet_resize] flat folder")
    return FlatImageDataset(str(root), transform=transform)


def ds_lsun_resize(transform, data_root):
    """
    LSUN_resize — same two possible layouts as Imagenet_resize.
    """
    root = find_dir(data_root, "LSUN_resize", "lsun_resize", "LSUN_resize")
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        print(f"  [LSUN_resize] ImageFolder ({len(subdirs)} subdirs)")
        return datasets.ImageFolder(str(root), transform=transform)
    print(f"  [LSUN_resize] flat folder")
    return FlatImageDataset(str(root), transform=transform)


def ds_imagenet_1k(transform, data_root, max_samples=None):
    """
    ImageNet-1k from HuggingFace datasets.
    max_samples: if set, limit to first N samples (useful for quick testing)
    Returns a torch Dataset wrapper for feature extraction.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required for ImageNet-1k. "
            "Install with: pip install datasets"
        )

    class HFDatasetWrapper(Dataset):
        """Wrap HuggingFace Dataset for PyTorch DataLoader."""
        def __init__(self, hf_dataset, transform, max_samples=None):
            self.hf_dataset = hf_dataset
            self.transform = transform
            self.max_samples = min(max_samples, len(hf_dataset)) if max_samples else len(hf_dataset)

        def __len__(self):
            return self.max_samples

        def __getitem__(self, idx):
            if idx >= len(self.hf_dataset):
                raise IndexError(f"Index {idx} out of range for dataset with length {self.max_samples}")

            item = self.hf_dataset[idx]
            img = item['image']
            if not img.mode == 'RGB':
                img = img.convert('RGB')

            if self.transform:
                img = self.transform(img)

            # Return dummy label (0) for OOD
            return img, 0

    # Load validation set (50k images)
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=False)

    print(f"  [ImageNet-1k] Loaded {len(ds)} validation images")
    if max_samples:
        print(f"  [ImageNet-1k] Limiting to first {max_samples} samples")

    return HFDatasetWrapper(ds, transform, max_samples)


def ds_dtd(split, transform, data_root):
    """
    DTD layout: /data/DTD/dtd/images/<category>/<img>.jpg
    torchvision.datasets.DTD (>=0.12) expects the *parent* of dtd/ as root.
    Falls back to ImageFolder on images/ for older torchvision.
    """
    root = find_dir(data_root, "DTD", "dtd")

    # Determine whether root IS dtd/ or its parent
    inner = root / "dtd"
    dtd_root = str(root.parent) if inner.exists() else str(root.parent)

    try:
        dset = datasets.DTD(root=dtd_root, split=split,
                            download=False, transform=transform)
        print(f"  [DTD] torchvision.datasets.DTD loaded (split={split})")
        return dset
    except AttributeError:
        # torchvision < 0.12
        img_dir = root / "dtd" / "images" if (root / "dtd").exists() \
                  else root / "images"
        print(f"  [DTD] older torchvision — ImageFolder on {img_dir}")
        return datasets.ImageFolder(str(img_dir), transform=transform)


def ds_places365(split, transform, data_root):
    """
    Places365 — unzip places365.zip first if not done:
        cd /home/tanmoy/research/data && unzip places365.zip -d places365

    Tries common extracted subfolder names, then torchvision loader.
    """
    try:
        root = find_dir(data_root, "places365", "Places365", "places365_standard")
    except FileNotFoundError:
        raise FileNotFoundError(
            "places365/ directory not found. "
            "Did you unzip places365.zip?\n"
            "  cd /home/tanmoy/research/data && unzip places365.zip -d places365"
        )

    # Try common subfolder layouts for the val set
    for candidate in ["val_256", "val", "test_256", "test", "data_256"]:
        cand = root / candidate
        if cand.is_dir():
            print(f"  [Places365] ImageFolder on {cand}")
            return datasets.ImageFolder(str(cand), transform=transform)

    # Try torchvision loader
    try:
        dset = datasets.Places365(str(root), split="val",
                                  small=True, download=False,
                                  transform=transform)
        print(f"  [Places365] torchvision.datasets.Places365 loaded")
        return dset
    except Exception as e:
        print(f"  [Places365] torchvision loader failed ({e}), trying ImageFolder")

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        print(f"  [Places365] ImageFolder on {root} ({len(subdirs)} subdirs)")
        return datasets.ImageFolder(str(root), transform=transform)

    return FlatImageDataset(str(root), transform=transform)


def ds_tinyimagenet(split, transform, data_root):
    """
    Tiny ImageNet — two possible val layouts:
      (A) val/<class>/<img>.JPEG  -> ImageFolder (class subfolders present)
      (B) val/images/<img>.JPEG + val_annotations.txt  -> TinyImageNetVal
    """
    # Try to find the base directory with train/ and val/ subdirs
    root = find_dir(data_root, "tiny-imagenet-200", "tiny_imagenet_200")

    # Handle nested structure: tiny-imagenet-200/tiny-imagenet-200/
    if (root / "tiny-imagenet-200" / split).is_dir():
        root = root / "tiny-imagenet-200"

    split_dir = root / split

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"tiny-imagenet-200/{split}/ not found at {root}"
        )

    if split == "val":
        subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
        # Layout A: class subdirs directly under val/
        if subdirs and subdirs[0].name != "images":
            print(f"  [TinyImageNet] ImageFolder layout (A)")
            return datasets.ImageFolder(str(split_dir), transform=transform)
        # Layout B: flat val/images/ + annotations
        print(f"  [TinyImageNet] annotations layout (B)")
        return TinyImageNetVal(str(root), transform=transform)
    else:
        return datasets.ImageFolder(str(split_dir), transform=transform)


# =============================================================================
# Custom Dataset classes
# =============================================================================

class FlatImageDataset(Dataset):
    """Images directly in a single folder (no class subdirs). label=0."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPEG", ".JPG", ".PNG"}

    def __init__(self, root, transform=None):
        self.files = sorted(
            p for p in Path(root).iterdir() if p.suffix in self.EXTS
        )
        if not self.files:
            raise FileNotFoundError(f"No images found in {root}")
        self.transform = transform
        print(f"  [FlatImageDataset] {len(self.files)} images in {root}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img) if self.transform else img, 0


class TinyImageNetVal(Dataset):
    """Handles flat val/images/ layout with val_annotations.txt."""

    def __init__(self, root, transform=None):
        root = Path(root)
        img_dir  = root / "val" / "images"
        ann_file = root / "val" / "val_annotations.txt"

        classes = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
        cls2idx = {c: i for i, c in enumerate(classes)}

        self.samples = []
        if ann_file.exists():
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    fname, cls = parts[0], parts[1]
                    p = img_dir / fname
                    if p.exists():
                        self.samples.append((str(p), cls2idx.get(cls, 0)))
        else:
            for p in sorted(img_dir.glob("*.JPEG")):
                self.samples.append((str(p), 0))

        self.transform = transform
        print(f"  [TinyImageNetVal] {len(self.samples)} images")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img, label


# =============================================================================
# Job table: CLI dataset name -> list of (job_key, builder_fn, output_name)
# =============================================================================

def make_jobs(data_root, transform, max_samples_imagenet=None):
    """Returns dict: job_key -> (dataset, output_filename)"""
    return {
        "cifar10_train":    (lambda: ds_cifar10("train",  transform, data_root), "cifar10_train.npz"),
        "cifar10_test":     (lambda: ds_cifar10("test",   transform, data_root), "cifar10_test.npz"),
        "cifar100_train":   (lambda: ds_cifar100("train", transform, data_root), "cifar100_train.npz"),
        "cifar100_test":    (lambda: ds_cifar100("test",  transform, data_root), "cifar100_test.npz"),
        "svhn_test":        (lambda: ds_svhn("test",      transform, data_root), "svhn_test.npz"),
        "imagenet_resize":  (lambda: ds_imagenet_resize(  transform, data_root), "imagenet_resize.npz"),
        "lsun_resize":      (lambda: ds_lsun_resize(      transform, data_root), "lsun_resize.npz"),
        "dtd_test":         (lambda: ds_dtd("test",       transform, data_root), "dtd_test.npz"),
        "places365_val":    (lambda: ds_places365("val",  transform, data_root), "places365_val.npz"),
        "tinyimagenet_val": (lambda: ds_tinyimagenet("val", transform, data_root), "tinyimagenet_val.npz"),
        "imagenet_1k_val":  (lambda: ds_imagenet_1k(transform, data_root, max_samples=max_samples_imagenet),
                             "imagenet_1k_val.npz"),
    }


# CLI name -> list of job keys
DATASET_JOBS = {
    "cifar10":          ["cifar10_train",   "cifar10_test"],
    "cifar100":         ["cifar100_train",  "cifar100_test"],
    "svhn":             ["svhn_test"],
    "imagenet_resize":  ["imagenet_resize"],
    "imagenet_1k":      ["imagenet_1k_val"],
    "lsun_resize":      ["lsun_resize"],
    "dtd":              ["dtd_test"],
    "places365":        ["places365_val"],
    "tinyimagenet":     ["tinyimagenet_val"],
}


# =============================================================================
# Core extraction
# =============================================================================

@torch.no_grad()
def extract_features(model, loader, device, l2_normalize=True):
    all_feats, all_labels = [], []
    # Determine model dtype to cast inputs appropriately
    model_dtype = next(model.parameters()).dtype
    for images, labels in tqdm(loader, desc="  extracting", leave=False):
        # Convert images to match model dtype
        images = images.to(device, non_blocking=True, dtype=model_dtype)
        feats = model(images)
        if l2_normalize:
            feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().float().numpy())
        all_labels.append(
            labels.numpy() if isinstance(labels, torch.Tensor)
            else np.array(labels)
        )
    return np.concatenate(all_feats), np.concatenate(all_labels)


def save_npz(path, features, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), features=features, labels=labels)
    print(f"  -> {path.name}  shape={features.shape}  "
          f"{path.stat().st_size/1e6:.1f} MB")


# =============================================================================
# Main runner
# =============================================================================

def run_arch(arch_name, model, preprocess, job_keys,
             data_root, cache_dir, device, resume, max_samples_imagenet=None):

    arch_dir = Path(cache_dir) / arch_name
    arch_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = make_jobs(data_root, preprocess, max_samples_imagenet=max_samples_imagenet)

    for key in job_keys:
        print(f"\n[{arch_name}] {key}")
        builder_fn, fname = all_jobs[key]

        out = arch_dir / fname
        if resume and out.exists():
            data = np.load(out)
            print(f"  SKIP (exists) shape={data['features'].shape}")
            continue

        try:
            dset = builder_fn()
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=False)
        t0 = time.time()
        features, labels = extract_features(model, loader, device)
        save_npz(out, features, labels)
        norms = np.linalg.norm(features, axis=1)
        print(f"  {time.time()-t0:.1f}s | "
              f"mean_norm={norms.mean():.4f} ± {norms.std():.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default=DATA_ROOT)
    p.add_argument("--cache_dir",  default=CACHE_DIR)
    p.add_argument("--models",     nargs="+", default=["clip", "dinov2"],
                   choices=["clip", "dinov2"])
    p.add_argument("--datasets",   nargs="+",
                   default=list(DATASET_JOBS.keys()),
                   choices=list(DATASET_JOBS.keys()))
    p.add_argument("--resume",     action="store_true",
                   help="Skip .npz files that already exist")
    p.add_argument("--use_bf16",   action="store_true",
                   help="bfloat16 inference — recommended on A4500 Ampere")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--max_samples_imagenet", type=int, default=None,
                   help="Limit ImageNet-1k to N samples (useful for testing)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    job_keys = []
    for d in args.datasets:
        job_keys.extend(DATASET_JOBS[d])

    # ── Print run config ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU      : {props.name}  ({props.total_memory/1e9:.1f} GB)")
    print(f"  bf16     : {args.use_bf16}")
    print(f"  batch    : {BATCH_SIZE}")
    print(f"  data     : {args.data_root}")
    print(f"  cache    : {args.cache_dir}")
    print(f"  jobs     : {job_keys}")
    print("="*60)

    # ── Places365 pre-flight check ────────────────────────────────────────────
    if "places365" in args.datasets:
        p365 = Path(args.data_root) / "places365"
        if not p365.exists():
            print("\n  WARNING: places365/ not found.")
            print("  Unzip first:")
            print(f"    cd {args.data_root} && unzip places365.zip -d places365\n")

    model_loaders = {
        "clip":   (load_clip_vitg14,   "clip_vitg14"),
        "dinov2": (load_dinov2_vitg14, "dinov2_vitg14"),
    }

    for model_key in args.models:
        loader_fn, arch_name = model_loaders[model_key]
        print(f"\n{'='*60}\n  Loading {arch_name} ...\n{'='*60}")

        model, preprocess = loader_fn(device)
        if args.use_bf16:
            model = model.half()

        run_arch(
            arch_name          = arch_name,
            model              = model,
            preprocess         = preprocess,
            job_keys           = job_keys,
            data_root          = args.data_root,
            cache_dir          = args.cache_dir,
            device             = device,
            resume             = args.resume,
            max_samples_imagenet = args.max_samples_imagenet,
        )
        del model
        torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nCache summary:\n")
    cache = Path(args.cache_dir)
    if cache.exists():
        for p in sorted(cache.rglob("*.npz")):
            d = np.load(p)
            rel = p.relative_to(cache)
            print(f"  {str(rel):<50} "
                  f"shape={str(d['features'].shape):<18} "
                  f"{p.stat().st_size/1e6:.1f} MB")
    print()


if __name__ == "__main__":
    main()