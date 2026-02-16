"""
quick_test.py
Quick test with CLIP and DINOv2 only (no WRN training required)

This will extract features using zero-shot foundation models and run evaluation.
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract_features import (
    CLIPExtractor, DINOv2Extractor,
    get_cifar10_loader, get_cifar100_loader, get_svhn_loader
)

def main():
    print("="*60)
    print("Quick Test: CLIP + DINOv2 Feature Extraction")
    print("="*60)

    # Create extractors
    print("\nLoading models...")
    clip_extractor = CLIPExtractor(model_name='ViT-L-14', pretrained='openai')
    dinov2_extractor = DINOv2Extractor(model_name='dinov2_vitl14')

    # Get dataloaders
    print("\nLoading datasets...")
    cifar10_test = get_cifar10_loader(train=False)
    cifar100_test = get_cifar100_loader(train=False)
    svhn_test = get_svhn_loader(split='test')

    # Extract CLIP features
    print("\n" + "="*60)
    print("Extracting CLIP features...")
    print("="*60)

    for name, loader in [("cifar10", cifar10_test), ("cifar100", cifar100_test), ("svhn", svhn_test)]:
        print(f"\nExtracting {name}...")
        features, labels = clip_extractor.extract(loader, max_samples=1000)  # Start with 1000 samples

        # Save
        save_path = Path(f'./features/{name}_test_clip.npz')
        save_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(save_path, features=features, labels=labels)
        print(f"Saved {len(features)} features to {save_path}")

    # Extract DINOv2 features
    print("\n" + "="*60)
    print("Extracting DINOv2 features...")
    print("="*60)

    for name, loader in [("cifar10", cifar10_test), ("cifar100", cifar100_test), ("svhn", svhn_test)]:
        print(f"\nExtracting {name}...")
        features, labels = dinov2_extractor.extract(loader, max_samples=1000)

        # Save
        save_path = Path(f'./features/{name}_test_dinov2.npz')
        save_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(save_path, features=features, labels=labels)
        print(f"Saved {len(features)} features to {save_path}")

    print("\n" + "="*60)
    print("Feature extraction complete!")
    print("="*60)
    print("\nNow run evaluation:")
    print("  python compute_metrics_and_eval.py")

if __name__ == '__main__':
    main()
