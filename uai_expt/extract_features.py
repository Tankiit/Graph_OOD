"""
extract_features.py
Unified feature extraction across architectures for UAI experiments.

This script extracts features from multiple architectures:
- WRN-40-2 from SAL/WOODS pretrained checkpoints
- CLIP (ViT-L-14) zero-shot
- DINOv2 (ViT-L) zero-shot

For datasets:
- In-distribution: CIFAR-10, CIFAR-100
- Out-of-distribution: SVHN, Textures, LSUN-C
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wrn import WideResNet

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

# Data root
DATA_ROOT = Path('/Users/tanmoy/research/data')

# Feature cache directory
CACHE_DIR = Path('./features')
CACHE_DIR.mkdir(exist_ok=True)

# Device
DEVICE = 'mps'  # Use MPS (Mac GPU) for faster processing

# ============================================================
# Dataset Loaders
# ============================================================

def get_cifar10_loader(batch_size=128, train=False):
    """Get CIFAR-10 data loader"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=train, download=True, transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_cifar100_loader(batch_size=128, train=False):
    """Get CIFAR-100 data loader"""
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    dataset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=train, download=True, transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_svhn_loader(batch_size=128, split='test'):
    """Get SVHN data loader"""
    normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    dataset = torchvision.datasets.SVHN(
        root=DATA_ROOT, split=split, download=True, transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_textures_loader(batch_size=128):
    """
    Get Textures (Describable Textures Dataset) data loader
    Download from: https://www.robots.ox.ac.uk/~vgg/data/dtd/
    """
    # Check if dataset exists
    dtd_path = DATA_ROOT / 'dtd' / 'images'
    if not dtd_path.exists():
        logger.error(f"DTD dataset not found at {dtd_path}")
        logger.error("Please download from: https://www.robots.ox.ac.uk/~vgg/data/dtd/")
        logger.error("Extract to: {DATA_ROOT}/dtd/")
        return None

    from torchvision.datasets import ImageFolder
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize
    ])

    dataset = ImageFolder(root=dtd_path, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_lsun_c_loader(batch_size=128):
    """
    Get LSUN-C (cropped LSUN) data loader
    Download from: https://github.com/yuval-alaluf/supervised-modeling-ood
    """
    # Check if dataset exists
    lsun_path = DATA_ROOT / 'lsun_c'
    if not lsun_path.exists():
        logger.error(f"LSUN-C dataset not found at {lsun_path}")
        logger.error("Please download from: https://github.com/yuval-alaluf/supervised-modeling-ood")
        logger.error("Extract to: {DATA_ROOT}/lsun_c/")
        return None

    from torchvision.datasets import ImageFolder
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize
    ])

    dataset = ImageFolder(root=lsun_path, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# ============================================================
# Feature Extractors
# ============================================================

class WRNExtractor:
    """WideResNet feature extractor for SAL experiments"""

    def __init__(self, checkpoint_path, depth=40, widen_factor=2, num_classes=10):
        self.checkpoint_path = checkpoint_path
        self.depth = depth
        self.widen_factor = widen_factor
        self.num_classes = num_classes

        # Load model
        logger.info(f"Loading WRN-{depth}-{widen_factor} from {checkpoint_path}")
        self.model = WideResNet(
            depth=depth,
            widen_factor=widen_factor,
            num_classes=num_classes,
            dropRate=0.0
        )

        # Load checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Remove 'module.' prefix if present (DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            logger.info("Loaded checkpoint successfully")
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}, using random initialization")

        self.model = self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def extract(self, dataloader, max_samples=None):
        """Extract penultimate layer features"""
        features_list = []
        labels_list = []

        for x, y in tqdm(dataloader, desc="Extracting WRN features"):
            x = x.to(DEVICE)

            # Extract features before final layer
            feat = self.model.extract_features(x)

            features_list.append(feat.cpu().numpy())
            labels_list.append(y.numpy())

            if max_samples and len(features_list) * dataloader.batch_size >= max_samples:
                break

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        if max_samples:
            features = features[:max_samples]
            labels = labels[:max_samples]

        return features, labels


class CLIPExtractor:
    """CLIP feature extractor (zero-shot)"""

    def __init__(self, model_name='ViT-L-14', pretrained='openai'):
        logger.info(f"Loading CLIP {model_name} ({pretrained})")

        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except ImportError:
            logger.error("open_clip not installed. Install with: pip install open_clip_torch")
            raise

        self.model = self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def extract(self, dataloader, max_samples=None):
        """Extract CLIP image features"""
        features_list = []
        labels_list = []

        for x, y in tqdm(dataloader, desc="Extracting CLIP features"):
            x = x.to(DEVICE)

            # Resize to 224x224 for CLIP
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            # Encode image
            feat = self.model.encode_image(x)

            # L2 normalize
            feat = feat / feat.norm(dim=-1, keepdim=True)

            features_list.append(feat.cpu().numpy())
            labels_list.append(y.numpy())

            if max_samples and len(features_list) * dataloader.batch_size >= max_samples:
                break

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        if max_samples:
            features = features[:max_samples]
            labels = labels[:max_samples]

        return features, labels


class DINOv2Extractor:
    """DINOv2 feature extractor (zero-shot)"""

    def __init__(self, model_name='dinov2_vitl14'):
        logger.info(f"Loading DINOv2 {model_name}")

        # Use torch.hub to load DINOv2
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def extract(self, dataloader, max_samples=None):
        """Extract DINOv2 features"""
        features_list = []
        labels_list = []

        for x, y in tqdm(dataloader, desc="Extracting DINOv2 features"):
            x = x.to(DEVICE)

            # Resize to 224x224 for DINOv2
            x_resized = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            # Extract features
            feat = self.model(x_resized)

            features_list.append(feat.cpu().numpy())
            labels_list.append(y.numpy())

            if max_samples and len(features_list) * dataloader.batch_size >= max_samples:
                break

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        if max_samples:
            features = features[:max_samples]
            labels = labels[:max_samples]

        return features, labels


# ============================================================
# Main Extraction Pipeline
# ============================================================

def extract_and_save(dataset_name, extractor, loader, split='test'):
    """Extract features and save to disk"""
    logger.info(f"\nExtracting {dataset_name} ({split})...")

    # Extract features
    features, labels = extractor.extract(loader)

    # Save to disk
    arch_name = extractor.__class__.__name__.replace('Extractor', '').lower()
    save_path = CACHE_DIR / f'{dataset_name}_{split}_{arch_name}.npz'

    np.savez_compressed(save_path, features=features, labels=labels)

    logger.info(f"Saved {len(features)} features to {save_path}")
    logger.info(f"Feature shape: {features.shape}")

    return features, labels


def main():
    """Main extraction pipeline"""

    # ============================================================
    # Step 1: WRN from SAL
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Extracting WRN Features")
    logger.info("="*60)

    # CIFAR-10 WRN checkpoint path (update this path)
    wrn_cifar10_path = './checkpoints/wrn40_2_cifar10.pt'

    # Create WRN extractor for CIFAR-10
    wrn_cifar10 = WRNExtractor(wrn_cifar10_path, depth=40, widen_factor=2, num_classes=10)

    # Extract features for CIFAR-10
    cifar10_test = get_cifar10_loader(train=False)
    extract_and_save('cifar10', wrn_cifar10, cifar10_test, split='test')

    # Extract for CIFAR-100 (need WRN-40-2 trained on CIFAR-100)
    wrn_cifar100_path = './checkpoints/wrn40_2_cifar100.pt'
    wrn_cifar100 = WRNExtractor(wrn_cifar100_path, depth=40, widen_factor=2, num_classes=100)

    cifar100_test = get_cifar100_loader(train=False)
    extract_and_save('cifar100', wrn_cifar100, cifar100_test, split='test')

    # Extract for OOD datasets
    svhn_test = get_svhn_loader(split='test')
    extract_and_save('svhn', wrn_cifar10, svhn_test, split='test')

    textures = get_textures_loader()
    if textures:
        extract_and_save('textures', wrn_cifar10, textures, split='test')

    lsun_c = get_lsun_c_loader()
    if lsun_c:
        extract_and_save('lsun_c', wrn_cifar10, lsun_c, split='test')

    # ============================================================
    # Step 2: CLIP Features
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Extracting CLIP Features")
    logger.info("="*60)

    clip_extractor = CLIPExtractor(model_name='ViT-L-14', pretrained='openai')

    # Extract for all datasets
    extract_and_save('cifar10', clip_extractor, cifar10_test, split='test')
    extract_and_save('cifar100', clip_extractor, cifar100_test, split='test')
    extract_and_save('svhn', clip_extractor, svhn_test, split='test')

    if textures:
        extract_and_save('textures', clip_extractor, textures, split='test')
    if lsun_c:
        extract_and_save('lsun_c', clip_extractor, lsun_c, split='test')

    # ============================================================
    # Step 3: DINOv2 Features
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Extracting DINOv2 Features")
    logger.info("="*60)

    dinov2_extractor = DINOv2Extractor(model_name='dinov2_vitl14')

    # Extract for all datasets
    extract_and_save('cifar10', dinov2_extractor, cifar10_test, split='test')
    extract_and_save('cifar100', dinov2_extractor, cifar100_test, split='test')
    extract_and_save('svhn', dinov2_extractor, svhn_test, split='test')

    if textures:
        extract_and_save('textures', dinov2_extractor, textures, split='test')
    if lsun_c:
        extract_and_save('lsun_c', dinov2_extractor, lsun_c, split='test')

    logger.info("\n" + "="*60)
    logger.info("Feature extraction complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
