"""
Quick test to verify WRN, CLIP, and DINOv2 extraction works
Extracts only 100 samples per dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract_features import (
    WRNExtractor, CLIPExtractor, DINOv2Extractor,
    get_cifar10_loader, get_svhn_loader
)
import numpy as np

def main():
    print("="*60)
    print("Testing Feature Extraction (100 samples)")
    print("="*60)

    # Get a small dataloader
    cifar10_test = get_cifar10_loader(train=False)
    svhn_test = get_svhn_loader(split='test')

    # Test WRN
    print("\n1. Testing WRN...")
    wrn_extractor = WRNExtractor(
        './checkpoints/wrn40_2_cifar10.pt',
        depth=40,
        widen_factor=2,
        num_classes=10
    )

    features, labels = wrn_extractor.extract(cifar10_test, max_samples=100)
    print(f"✓ WRN: Extracted {len(features)} features, shape={features.shape}")

    # Save test
    Path('./features').mkdir(exist_ok=True)
    np.savez_compressed('./features/test_wrn.npz', features=features, labels=labels)
    print("  Saved to features/test_wrn.npz")

    # Test CLIP
    print("\n2. Testing CLIP...")
    clip_extractor = CLIPExtractor(model_name='ViT-L-14', pretrained='openai')
    features, labels = clip_extractor.extract(cifar10_test, max_samples=100)
    print(f"✓ CLIP: Extracted {len(features)} features, shape={features.shape}")

    np.savez_compressed('./features/test_clip.npz', features=features, labels=labels)
    print("  Saved to features/test_clip.npz")

    # Test DINOv2
    print("\n3. Testing DINOv2...")
    dinov2_extractor = DINOv2Extractor(model_name='dinov2_vitl14')
    features, labels = dinov2_extractor.extract(cifar10_test, max_samples=100)
    print(f"✓ DINOv2: Extracted {len(features)} features, shape={features.shape}")

    np.savez_compressed('./features/test_dinov2.npz', features=features, labels=labels)
    print("  Saved to features/test_dinov2.npz")

    # Test SVHN with CLIP
    print("\n4. Testing SVHN with CLIP...")
    features, labels = clip_extractor.extract(svhn_test, max_samples=100)
    print(f"✓ CLIP SVHN: Extracted {len(features)} features, shape={features.shape}")

    np.savez_compressed('./features/test_svhn_clip.npz', features=features, labels=labels)
    print("  Saved to features/test_svhn_clip.npz")

    print("\n" + "="*60)
    print("✓ All extraction tests passed!")
    print("="*60)
    print("\nReady to run full extraction:")
    print("  python extract_features.py")

if __name__ == '__main__':
    main()
