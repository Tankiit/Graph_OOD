"""
Complete Integration: CIFAR Training + Energy-based OOD + Spectral Analysis + Attention Enhancement

This integrates:
1. CIFAR training pipeline
2. Energy-based OOD detection
3. Spectral analysis with k-NN adjacency matrices
4. Attention-based graph enhancements (optional)

File structure for your repo:
graph-spectral-ood/CIFAR/
├── train.py (this file)
├── models/
│   ├── wrn.py
│   └── resnet.py
├── dataloader/
│   ├── cifar10.py
│   └── cifar100.py
├── utils/
│   ├── spectral_monitor.py (new)
│   └── wang_li_energy.py (new)
└── ood_postprocessors/
    └── spectral_schemanet.py (new)
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph
import faiss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: ENERGY-BASED OOD DETECTION
# ============================================================================

class EnergyOODDetector:
    """
    Energy-based OOD Detection
    Reference: Energy-based Out-of-distribution Detection (NeurIPS 2020)
    """

    def __init__(self, model, temperature=1.0):
        self.model = model
        self.temperature = temperature

    def compute_energy_scores(self, dataloader, device='cuda'):
        """
        Compute energy scores: E(x) = -T * log(sum(exp(f(x)/T)))
        Lower energy = more OOD
        """
        self.model.eval()
        scores = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch

                data = data.to(device)
                logits = self.model(data)

                # Energy score
                energy = -self.temperature * torch.logsumexp(
                    logits / self.temperature, dim=1
                )
                scores.extend(energy.cpu().numpy())

        return np.array(scores)

    def extract_features(self, dataloader, device='cuda'):
        """Extract penultimate layer features for graph construction"""
        self.model.eval()
        features = []
        labels = []

        # Register hook to capture features
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0] if isinstance(input, tuple) else input
            return hook

        # Hook before final linear layer
        if hasattr(self.model, 'fc'):
            handle = self.model.fc.register_forward_hook(get_activation('features'))
        elif hasattr(self.model, 'linear'):
            handle = self.model.linear.register_forward_hook(get_activation('features'))
        else:
            raise ValueError("Model doesn't have fc or linear layer")

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data, batch_labels = batch[0], batch[1]
                else:
                    data, batch_labels = batch, None

                data = data.to(device)
                _ = self.model(data)

                feat = activation['features'].cpu().numpy()
                features.append(feat)
                if batch_labels is not None:
                    labels.extend(batch_labels.cpu().numpy())

        handle.remove()

        features = np.vstack(features)
        labels = np.array(labels) if labels else None

        return features, labels


# ============================================================================
# PART 2: SPECTRAL ANALYSIS WITH K-NN ADJACENCY
# ============================================================================

class SpectralOODDetector:
    """
    Spectral OOD detection using k-NN graph adjacency matrices
    Monitors spectral gap changes when adding test samples
    """

    def __init__(self, k_neighbors=50, n_eigenvalues=10):
        self.k = k_neighbors
        self.n_eigenvalues = n_eigenvalues
        self.train_features = None
        self.baseline_lambda2 = None
        self.baseline_eigenvalues = None

    def build_knn_adjacency(self, features):
        """
        Build k-NN adjacency matrix using FAISS
        Returns symmetric weighted adjacency matrix
        """
        n_samples = features.shape[0]

        # Normalize features
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

        # Build FAISS index
        index = faiss.IndexFlatL2(features_norm.shape[1])
        index.add(features_norm.astype(np.float32))

        # Search k+1 neighbors (including self)
        k_search = min(self.k + 1, n_samples)
        D, I = index.search(features_norm.astype(np.float32), k_search)

        # Remove self-connections
        D = D[:, 1:]
        I = I[:, 1:]

        # Compute similarity weights (inverse distance)
        similarities = 1.0 / (1.0 + D)

        # Build adjacency matrix
        adj_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            adj_matrix[i, I[i]] = similarities[i]

        # Symmetrize
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

        return adj_matrix

    def compute_spectral_gap(self, adjacency):
        """
        Compute spectral gap from adjacency matrix
        Returns λ₂ (second smallest eigenvalue of normalized Laplacian)
        """
        # Compute normalized Laplacian
        laplacian = csgraph.laplacian(adjacency, normed=True)

        # Compute eigenvalues
        try:
            n_eigs = min(self.n_eigenvalues, laplacian.shape[0] - 2)
            if n_eigs < 2:
                return 0.0, np.zeros(self.n_eigenvalues)

            eigenvalues = np.linalg.eigvalsh(laplacian)
            eigenvalues = np.sort(eigenvalues)[:self.n_eigenvalues]

            lambda2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0

            return lambda2, eigenvalues

        except Exception as e:
            print(f"Eigenvalue computation failed: {e}")
            return 0.0, np.zeros(self.n_eigenvalues)

    def fit(self, train_features):
        """Fit on training data to establish baseline"""
        print("Building training graph for spectral baseline...")
        self.train_features = train_features

        # Build adjacency matrix
        adj_matrix = self.build_knn_adjacency(train_features)

        # Compute baseline spectral properties
        self.baseline_lambda2, self.baseline_eigenvalues = self.compute_spectral_gap(adj_matrix)

        print(f"Baseline λ₂: {self.baseline_lambda2:.6f}")

    def score_samples(self, test_features):
        """
        Score test samples by measuring spectral gap deviation
        Higher score = more OOD
        """
        if self.train_features is None:
            raise ValueError("Must fit on training data first")

        scores = []

        for i in range(len(test_features)):
            # Add test sample to training graph
            combined_features = np.vstack([
                self.train_features,
                test_features[i:i+1]
            ])

            # Build augmented graph
            adj_matrix = self.build_knn_adjacency(combined_features)

            # Compute new spectral gap
            lambda2_new, _ = self.compute_spectral_gap(adj_matrix)

            # Score is deviation from baseline
            score = abs(lambda2_new - self.baseline_lambda2)
            scores.append(score)

            if (i + 1) % 100 == 0:
                print(f"  Scored {i+1}/{len(test_features)} samples")

        return np.array(scores)


# ============================================================================
# PART 3: SCHEMANET ATTENTION ENHANCEMENT (Optional)
# ============================================================================

class AttentionEnhancedAdjacency:
    """
    Enhance k-NN adjacency with ViT attention patterns
    Optional - requires ViT model
    """

    def __init__(self, vit_model=None, beta1=0.7, beta2=0.3):
        self.vit_model = vit_model
        self.beta1 = beta1  # Attention weight
        self.beta2 = beta2  # Spatial weight

    def enhance_adjacency(self, knn_adj, images=None):
        """Add attention information to k-NN adjacency"""
        if self.vit_model is None or images is None:
            return knn_adj

        try:
            with torch.no_grad():
                outputs = self.vit_model(images, output_attentions=True)
                attention = outputs.attentions[-1].mean(dim=1)
                attention = attention[:, 1:, 1:]  # Remove CLS

            # Average over batch
            attention_adj = attention.mean(dim=0).cpu().numpy()

            # Blend with k-NN
            if attention_adj.shape[0] == knn_adj.shape[0]:
                enhanced = 0.7 * knn_adj + 0.3 * attention_adj
            else:
                enhanced = knn_adj

            return enhanced

        except Exception as e:
            print(f"Attention enhancement failed: {e}")
            return knn_adj


# ============================================================================
# PART 4: INTEGRATED OOD EVALUATION
# ============================================================================

def evaluate_ood_methods(model, train_loader, test_id_loader, test_ood_loader,
                        device='cuda', use_attention=False, vit_model=None):
    """
    Comprehensive OOD evaluation comparing all methods

    Returns dict with results from:
    - Energy-based method
    - Spectral k-NN method
    - Combined ensemble method
    """
    print("="*70)
    print("COMPREHENSIVE OOD DETECTION EVALUATION")
    print("="*70)

    results = {}

    # Initialize detectors
    energy_detector = EnergyOODDetector(model, temperature=1.0)
    spectral_detector = SpectralOODDetector(k_neighbors=50, n_eigenvalues=10)

    # Extract features
    print("\n1. Extracting features...")
    train_features, _ = energy_detector.extract_features(train_loader, device)
    test_id_features, _ = energy_detector.extract_features(test_id_loader, device)
    test_ood_features, _ = energy_detector.extract_features(test_ood_loader, device)

    print(f"   Train features: {train_features.shape}")
    print(f"   Test ID features: {test_id_features.shape}")
    print(f"   Test OOD features: {test_ood_features.shape}")

    # Method 1: Energy-based Detection
    print("\n2. Testing Energy-based Method...")
    start_time = time.time()
    energy_scores_id = energy_detector.compute_energy_scores(test_id_loader, device)
    energy_scores_ood = energy_detector.compute_energy_scores(test_ood_loader, device)
    energy_time = time.time() - start_time

    # Energy: lower = more OOD, so negate for evaluation
    energy_labels = np.concatenate([np.zeros(len(energy_scores_id)),
                                   np.ones(len(energy_scores_ood))])
    energy_scores = np.concatenate([energy_scores_id, energy_scores_ood])
    energy_auc = roc_auc_score(energy_labels, -energy_scores)
    energy_ap = average_precision_score(energy_labels, -energy_scores)

    print(f"   Energy AUC: {energy_auc:.4f}")
    print(f"   Energy AP: {energy_ap:.4f}")
    print(f"   Time: {energy_time:.3f}s")

    results['energy'] = {
        'auc': energy_auc,
        'ap': energy_ap,
        'time': energy_time,
        'scores_id': energy_scores_id,
        'scores_ood': energy_scores_ood
    }

    # Method 2: Spectral k-NN
    print("\n3. Testing Spectral k-NN Method...")

    # Use subset of training data for efficiency
    n_train_subset = min(2000, len(train_features))
    train_subset_idx = np.random.choice(len(train_features), n_train_subset, replace=False)
    train_features_subset = train_features[train_subset_idx]

    spectral_detector.fit(train_features_subset)

    start_time = time.time()
    spectral_scores_id = spectral_detector.score_samples(test_id_features)
    spectral_scores_ood = spectral_detector.score_samples(test_ood_features)
    spectral_time = time.time() - start_time

    spectral_labels = np.concatenate([np.zeros(len(spectral_scores_id)),
                                     np.ones(len(spectral_scores_ood))])
    spectral_scores = np.concatenate([spectral_scores_id, spectral_scores_ood])
    spectral_auc = roc_auc_score(spectral_labels, spectral_scores)
    spectral_ap = average_precision_score(spectral_labels, spectral_scores)

    print(f"   Spectral AUC: {spectral_auc:.4f}")
    print(f"   Spectral AP: {spectral_ap:.4f}")
    print(f"   Time: {spectral_time:.3f}s")

    results['spectral'] = {
        'auc': spectral_auc,
        'ap': spectral_ap,
        'time': spectral_time,
        'scores_id': spectral_scores_id,
        'scores_ood': spectral_scores_ood,
        'baseline_lambda2': spectral_detector.baseline_lambda2
    }

    # Method 3: Combined (Ensemble)
    print("\n4. Testing Combined Method...")

    # Normalize scores to [0, 1]
    energy_scores_norm = (-energy_scores - energy_scores.min()) / (energy_scores.max() - energy_scores.min() + 1e-10)
    spectral_scores_norm = (spectral_scores - spectral_scores.min()) / (spectral_scores.max() - spectral_scores.min() + 1e-10)

    # Weighted combination
    combined_scores = 0.5 * energy_scores_norm + 0.5 * spectral_scores_norm
    combined_auc = roc_auc_score(energy_labels, combined_scores)
    combined_ap = average_precision_score(energy_labels, combined_scores)

    print(f"   Combined AUC: {combined_auc:.4f}")
    print(f"   Combined AP: {combined_ap:.4f}")

    results['combined'] = {
        'auc': combined_auc,
        'ap': combined_ap,
        'scores': combined_scores
    }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Method          | AUC    | AP     | Time")
    print("-" * 70)
    print(f"Energy          | {energy_auc:.4f} | {energy_ap:.4f} | {energy_time:.3f}s")
    print(f"Spectral        | {spectral_auc:.4f} | {spectral_ap:.4f} | {spectral_time:.3f}s")
    print(f"Combined        | {combined_auc:.4f} | {combined_ap:.4f} | -")
    print("="*70)

    return results


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

def visualize_ood_results(results, save_path='ood_results.png'):
    """Create comprehensive visualization of OOD detection results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Energy score distributions
    ax = axes[0, 0]
    ax.hist(results['energy']['scores_id'], bins=50, alpha=0.6,
           label='ID', color='blue', density=True)
    ax.hist(results['energy']['scores_ood'], bins=50, alpha=0.6,
           label='OOD', color='red', density=True)
    ax.set_xlabel('Energy Score')
    ax.set_ylabel('Density')
    ax.set_title('Energy-based Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Spectral score distributions
    ax = axes[0, 1]
    ax.hist(results['spectral']['scores_id'], bins=50, alpha=0.6,
           label='ID', color='blue', density=True)
    ax.hist(results['spectral']['scores_ood'], bins=50, alpha=0.6,
           label='OOD', color='red', density=True)
    ax.set_xlabel('Spectral Gap Deviation')
    ax.set_ylabel('Density')
    ax.set_title('Spectral k-NN Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. ROC Curves
    from sklearn.metrics import roc_curve
    ax = axes[0, 2]

    # Energy ROC
    energy_labels = np.concatenate([
        np.zeros(len(results['energy']['scores_id'])),
        np.ones(len(results['energy']['scores_ood']))
    ])
    energy_scores = np.concatenate([
        results['energy']['scores_id'],
        results['energy']['scores_ood']
    ])
    fpr_e, tpr_e, _ = roc_curve(energy_labels, -energy_scores)
    ax.plot(fpr_e, tpr_e, label=f"Energy (AUC={results['energy']['auc']:.3f})",
           linewidth=2, color='orange')

    # Spectral ROC
    spectral_labels = np.concatenate([
        np.zeros(len(results['spectral']['scores_id'])),
        np.ones(len(results['spectral']['scores_ood']))
    ])
    spectral_scores = np.concatenate([
        results['spectral']['scores_id'],
        results['spectral']['scores_ood']
    ])
    fpr_s, tpr_s, _ = roc_curve(spectral_labels, spectral_scores)
    ax.plot(fpr_s, tpr_s, label=f"Spectral (AUC={results['spectral']['auc']:.3f})",
           linewidth=2, color='green')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. AUC Comparison
    ax = axes[1, 0]
    methods = ['Energy', 'Spectral', 'Combined']
    aucs = [results['energy']['auc'], results['spectral']['auc'],
           results['combined']['auc']]
    bars = ax.bar(methods, aucs, color=['orange', 'green', 'purple'], alpha=0.7)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Method Comparison')
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

    # 5. Computation time
    ax = axes[1, 1]
    times = [results['energy']['time'], results['spectral']['time']]
    bars = ax.bar(['Energy', 'Spectral'], times, color=['orange', 'green'], alpha=0.7)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Efficiency')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{t:.2f}s', ha='center', va='bottom')

    # 6. Summary metrics
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
Performance Summary:

Energy Method:
  • AUC: {results['energy']['auc']:.4f}
  • AP: {results['energy']['ap']:.4f}
  • Time: {results['energy']['time']:.2f}s

Spectral Method:
  • AUC: {results['spectral']['auc']:.4f}
  • AP: {results['spectral']['ap']:.4f}
  • Time: {results['spectral']['time']:.2f}s
  • λ₂ baseline: {results['spectral']['baseline_lambda2']:.6f}

Combined Method:
  • AUC: {results['combined']['auc']:.4f}
  • AP: {results['combined']['ap']:.4f}

Improvement: {(results['spectral']['auc'] - results['energy']['auc']):.4f}
Speedup: {(results['energy']['time']/results['spectral']['time']):.2f}x
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle('OOD Detection: Energy-based vs Spectral Methods',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")

    plt.show()

    return fig


# ============================================================================
# PART 6: MAIN TRAINING SCRIPT
# ============================================================================

def get_cifar_loaders(dataset='cifar10', batch_size=128, num_workers=4, data_dir='/Users/tanmoy/research/data'):
    """Create CIFAR data loaders"""

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

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        # SVHN as OOD
        oodset = torchvision.datasets.SVHN(
            root=data_dir, split='test', download=True, transform=transform_test
        )
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        # Use CIFAR10 as OOD
        oodset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )

    train_loader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    ood_loader = DataLoader(oodset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, ood_loader


def main():
    parser = argparse.ArgumentParser(description='CIFAR OOD Detection')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'wrn'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str, default='/Users/tanmoy/research/data',
                       help='Directory to store/load datasets')
    parser.add_argument('--eval_ood_every', type=int, default=50,
                       help='Evaluate OOD detection every N epochs')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to evaluate')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, no training')

    args = parser.parse_args()

    # Device detection with MPS support
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    train_loader, test_loader, ood_loader = get_cifar_loaders(
        args.dataset, args.batch_size, data_dir=args.data_dir
    )

    # Load model
    print(f"Loading {args.model} model...")
    if args.model == 'resnet18':
        from torchvision.models import resnet18
        num_classes = 10 if args.dataset == 'cifar10' else 100
        model = resnet18(num_classes=num_classes)
    elif args.model == 'resnet50':
        from torchvision.models import resnet50
        num_classes = 10 if args.dataset == 'cifar10' else 100
        model = resnet50(num_classes=num_classes)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    model = model.to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    if not args.eval_only:
        # Training loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                             momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        print("\nStarting training...")
        for epoch in range(args.epochs):
            # Train
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            acc = 100. * correct / total
            print(f'Epoch {epoch+1}/{args.epochs} | Loss: {train_loss/len(train_loader):.3f} | Acc: {acc:.2f}%')

            # Evaluate OOD detection periodically
            if (epoch + 1) % args.eval_ood_every == 0:
                print(f"\n{'='*70}")
                print(f"OOD Evaluation at Epoch {epoch+1}")
                print(f"{'='*70}")

                results = evaluate_ood_methods(
                    model, train_loader, test_loader, ood_loader, device
                )

                visualize_ood_results(
                    results,
                    save_path=f'ood_results_epoch_{epoch+1}.png'
                )

    else:
        # Evaluation only
        print("\nEvaluating OOD detection...")
        results = evaluate_ood_methods(
            model, train_loader, test_loader, ood_loader, device
        )

        visualize_ood_results(results, save_path='ood_results_final.png')

    print("\nDone!")


if __name__ == '__main__':
    main()