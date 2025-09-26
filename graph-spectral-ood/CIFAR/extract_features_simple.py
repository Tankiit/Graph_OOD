#!/usr/bin/env python
"""
Simplified feature extraction script for spectral OOD detection.
This version only extracts features from the main CIFAR-10 dataset without corruptions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import faiss
from scipy.sparse import csgraph, csr_matrix
import time
import pickle
from pathlib import Path
import os
import argparse
from sklearn.metrics import roc_auc_score

# Import models
import sys
sys.path.insert(0, 'models')
from wrn import WideResNet

class SpectralFeatureExtractor:
    """Extract and cache features for spectral OOD detection"""

    def __init__(self, cache_dir='./spectral_features'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.use_gpu = torch.cuda.is_available()

    def extract_features(self, model, dataloader, desc=""):
        """Extract features from model"""
        model.eval()
        features_list = []
        labels_list = []

        print(f"Extracting {desc} features...")
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                if self.use_gpu:
                    data = data.cuda()

                # Get features from penultimate layer
                features = self._get_penultimate_features(model, data)

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())

                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(dataloader)}")

        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        print(f"  Extracted features shape: {features.shape}")

        return features, labels

    def _get_penultimate_features(self, model, data):
        """Extract penultimate layer features using hooks"""
        features = []

        def hook(module, input, output):
            # Store the input to the final layer
            features.append(input[0] if isinstance(input, tuple) else input)

        # Register hook on final linear layer
        if hasattr(model, 'fc'):
            handle = model.fc.register_forward_hook(hook)
        elif hasattr(model, 'linear'):
            handle = model.linear.register_forward_hook(hook)
        else:
            # For WideResNet, the final layer is usually called 'linear'
            # Let's find it dynamically
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and 'linear' in name.lower():
                    handle = module.register_forward_hook(hook)
                    break
            else:
                raise ValueError("Cannot find final linear layer")

        # Forward pass
        _ = model(data)

        # Remove hook
        handle.remove()

        return features[0]

    def save_features(self, features, labels, filename):
        """Save features to cache"""
        filepath = self.cache_dir / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump({'features': features, 'labels': labels}, f)
        print(f"Saved features to {filepath}")

    def load_features(self, filename):
        """Load features from cache"""
        filepath = self.cache_dir / f"{filename}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded features from {filepath}")
            return data['features'], data['labels']
        return None, None

def get_cifar10_loaders(batch_size=128):
    """Get CIFAR-10 data loaders"""

    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_svhn_loader(batch_size=128):
    """Get SVHN data loader (OOD dataset)"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    svhn_test = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform
    )
    svhn_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=2)

    return svhn_loader

def compute_spectral_scores(train_features, test_features, k=20):
    """Compute spectral OOD scores"""

    print("\nComputing spectral scores...")

    # Normalize features
    train_features_norm = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
    test_features_norm = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)

    # Build FAISS index for k-NN search
    d = train_features_norm.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
    index.add(train_features_norm.astype(np.float32))

    # Compute scores for test samples
    scores = []
    for i in range(len(test_features_norm)):
        test_feat = test_features_norm[i:i+1].astype(np.float32)

        # Find k nearest neighbors
        similarities, indices = index.search(test_feat, k)

        # Compute connectivity score (higher connectivity = lower OOD score)
        connectivity = np.mean(similarities[0])
        score = 1 - connectivity  # Convert to OOD score
        scores.append(score)

    return np.array(scores)

def main():
    parser = argparse.ArgumentParser(description='Simple Feature Extraction for Spectral OOD')
    parser.add_argument('--load_pretrained', type=str, required=True,
                       help='Path to pretrained model')
    parser.add_argument('--cache_dir', type=str, default='./spectral_features_simple',
                       help='Directory to cache features')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--k', type=int, default=20,
                       help='Number of nearest neighbors for spectral analysis')
    args = parser.parse_args()

    # Create feature extractor
    extractor = SpectralFeatureExtractor(cache_dir=args.cache_dir)

    # Load model
    print(f"Loading model from: {args.load_pretrained}")
    net = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.3, args=args)

    if torch.cuda.is_available():
        checkpoint = torch.load(args.load_pretrained)
    else:
        checkpoint = torch.load(args.load_pretrained, map_location=torch.device('cpu'))

    net.load_state_dict(checkpoint)
    print("Model loaded successfully!")

    if torch.cuda.is_available():
        net = net.cuda()

    # Get data loaders
    print("\nLoading datasets...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    svhn_loader = get_svhn_loader(batch_size=args.batch_size)

    print(f"CIFAR-10 train size: {len(train_loader.dataset)}")
    print(f"CIFAR-10 test size: {len(test_loader.dataset)}")
    print(f"SVHN test size: {len(svhn_loader.dataset)}")

    # Extract features
    print("\n" + "="*60)
    print("EXTRACTING FEATURES")
    print("="*60)

    # Try to load cached features first
    train_features, train_labels = extractor.load_features('cifar10_train')
    if train_features is None:
        train_features, train_labels = extractor.extract_features(net, train_loader, "CIFAR-10 train")
        extractor.save_features(train_features, train_labels, 'cifar10_train')

    test_features, test_labels = extractor.load_features('cifar10_test')
    if test_features is None:
        test_features, test_labels = extractor.extract_features(net, test_loader, "CIFAR-10 test")
        extractor.save_features(test_features, test_labels, 'cifar10_test')

    svhn_features, svhn_labels = extractor.load_features('svhn_test')
    if svhn_features is None:
        svhn_features, svhn_labels = extractor.extract_features(net, svhn_loader, "SVHN test (OOD)")
        extractor.save_features(svhn_features, svhn_labels, 'svhn_test')

    # Compute spectral OOD scores
    print("\n" + "="*60)
    print("COMPUTING SPECTRAL OOD SCORES")
    print("="*60)

    # Compute scores
    test_scores = compute_spectral_scores(train_features, test_features, k=args.k)
    svhn_scores = compute_spectral_scores(train_features, svhn_features, k=args.k)

    # Evaluate OOD detection performance
    print("\n" + "="*60)
    print("OOD DETECTION RESULTS")
    print("="*60)

    # Combine scores and labels
    all_scores = np.concatenate([test_scores, svhn_scores])
    all_labels = np.concatenate([np.zeros(len(test_scores)), np.ones(len(svhn_scores))])

    # Compute AUROC
    auroc = roc_auc_score(all_labels, all_scores)
    print(f"\nSpectral OOD Detection AUROC: {auroc:.4f}")

    # Compute FPR95
    sorted_indices = np.argsort(test_scores)
    threshold_idx = int(0.95 * len(test_scores))
    threshold = test_scores[sorted_indices[threshold_idx]]
    fpr95 = np.mean(svhn_scores < threshold)
    print(f"FPR at 95% TPR: {fpr95:.4f}")

    # Save results
    results = {
        'test_scores': test_scores,
        'svhn_scores': svhn_scores,
        'auroc': auroc,
        'fpr95': fpr95,
        'k': args.k
    }

    results_path = Path(args.cache_dir) / 'spectral_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()