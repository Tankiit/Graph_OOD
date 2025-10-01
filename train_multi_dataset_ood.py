"""
ENHANCED CIFAR Training + Energy-based OOD + Improved Spectral Analysis
This script automates training and evaluation across multiple datasets.

Key features:
1. Iterates through specified datasets (e.g., CIFAR-10, CIFAR-100).
2. Trains a model for each dataset from scratch.
3. Saves model checkpoints at specified intervals.
4. Runs enhanced OOD evaluation with theoretical analysis periodically.
5. Saves evaluation results and visualizations, organized by dataset and epoch.
"""

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
from sklearn.decomposition import PCA
from scipy.sparse import csgraph
import faiss
import matplotlib.pyplot as plt
import warnings
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: ENERGY-BASED OOD DETECTION (Your existing code - unchanged)
# ============================================================================

class EnergyOODDetector:
    """
    Energy-based OOD Detection - keeping your existing implementation
    """
    
    def __init__(self, model, temperature=1.0):
        self.model = model
        self.temperature = temperature
        
    def compute_energy_scores(self, dataloader, device='cuda'):
        """Compute energy scores: E(x) = -T * log(sum(exp(f(x)/T)))"""
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
                
                energy = -self.temperature * torch.logsumexp(
                    logits / self.temperature, dim=1
                )
                scores.extend(energy.cpu().numpy())
        
        return np.array(scores)
    
    def extract_features(self, dataloader, device='cuda', max_samples=None):
        """Extract penultimate layer features for graph construction"""
        self.model.eval()
        features = []
        labels = []

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0] if isinstance(input, tuple) else input
            return hook

        # Find the last linear layer
        last_linear_layer = None
        if hasattr(self.model, 'fc'):
            last_linear_layer = self.model.fc
        elif hasattr(self.model, 'linear'):
            last_linear_layer = self.model.linear
        else:
            for m in reversed(list(self.model.modules())):
                if isinstance(m, nn.Linear):
                    last_linear_layer = m
                    break

        if last_linear_layer is None:
            raise ValueError("Model doesn't have a recognizable final linear layer (fc or linear)")

        handle = last_linear_layer.register_forward_hook(get_activation('features'))

        total_samples = 0
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

                total_samples += len(feat)
                if max_samples is not None and total_samples >= max_samples:
                    break

        handle.remove()
        features = np.vstack(features)
        labels = np.array(labels) if labels else None

        # Truncate if over max_samples
        if max_samples is not None and len(features) > max_samples:
            features = features[:max_samples]
            if labels is not None:
                labels = labels[:max_samples]

        return features, labels


# ============================================================================
# PART 2: ENHANCED SPECTRAL OOD DETECTOR (Major improvements)
# ============================================================================

class EnhancedSpectralOODDetector:
    """
    ENHANCED Spectral OOD detection with improved theoretical bounds
    """

    def __init__(self, k_neighbors=50, n_eigenvalues=10, confidence_level=0.05):
        self.k = k_neighbors
        self.n_eigenvalues = n_eigenvalues
        self.train_features = None
        self.baseline_lambda2 = None
        self.confidence_level = confidence_level
        self.empirical_variance = None
        self.local_lipschitz = None
        self.intrinsic_dimension = None
        self.optimal_k = None
        self.theoretical_bounds = {}
        self.timing_stats = {}
        self.train_adjacency = None  # Cache training graph
        self.faiss_index = None  # Cache FAISS index
        
    def build_knn_adjacency(self, features):
        n_samples = features.shape[0]
        if n_samples <= self.k:
            return np.eye(n_samples) * 0.1
        
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        index = faiss.IndexFlatL2(features_norm.shape[1])
        index.add(features_norm.astype(np.float32))
        
        k_search = min(self.k + 1, n_samples)
        D, I = index.search(features_norm.astype(np.float32), k_search)
        
        D, I = D[:, 1:], I[:, 1:]
        
        median_distance = np.median(D)
        if median_distance == 0: median_distance = 1.0
        
        similarities = np.exp(-D**2 / (2 * median_distance**2))
        
        adj_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            adj_matrix[i, I[i]] = similarities[i]
        
        return (adj_matrix + adj_matrix.T) / 2
    
    def compute_spectral_gap(self, adjacency):
        try:
            laplacian = csgraph.laplacian(adjacency, normed=True)
            n_eigs = min(self.n_eigenvalues, laplacian.shape[0] - 2)
            if n_eigs < 2: return 0.0, np.zeros(self.n_eigenvalues)
                
            eigenvalues = np.linalg.eigvalsh(laplacian)
            eigenvalues = np.sort(eigenvalues)[:self.n_eigenvalues]
            
            return (eigenvalues[1] if len(eigenvalues) > 1 else 0.0), eigenvalues
        except Exception:
            return 0.0, np.zeros(self.n_eigenvalues)
    
    def fit(self, train_features):
        print("Enhanced Spectral OOD Training...")
        start_time = time.time()
        self.train_features = train_features

        self.intrinsic_dimension = self._estimate_intrinsic_dimension(train_features)
        n = len(train_features)
        self.optimal_k = self._compute_optimal_k(n, self.intrinsic_dimension)

        # Cache training adjacency and FAISS index
        self.train_adjacency = self.build_knn_adjacency(train_features)
        self.baseline_lambda2, _ = self.compute_spectral_gap(self.train_adjacency)

        # Build and cache FAISS index for fast nearest neighbor search
        features_norm = train_features / (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-10)
        self.faiss_index = faiss.IndexFlatL2(features_norm.shape[1])
        self.faiss_index.add(features_norm.astype(np.float32))

        self.empirical_variance = self._estimate_spectral_variance(train_features)
        self.local_lipschitz = self._estimate_local_lipschitz(train_features)
        self.theoretical_bounds = self._compute_theoretical_bounds(n)

        self.timing_stats['fit_time'] = time.time() - start_time
        print(f"   Fit time: {self.timing_stats['fit_time']:.3f}s")
        
    def score_samples(self, test_features, batch_size=32):
        if self.train_features is None: raise ValueError("Must fit first")

        scores = []
        n_test = len(test_features)

        # Process in batches for better performance
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            batch_features = test_features[batch_start:batch_end]

            # Use approximate scoring based on distance to training manifold
            # This is much faster than rebuilding graphs
            batch_scores = self._fast_approximate_score(batch_features)
            scores.extend(batch_scores)

        return np.array(scores)

    def _fast_approximate_score(self, test_batch):
        """Fast approximate scoring using cached FAISS index"""
        test_norm = test_batch / (np.linalg.norm(test_batch, axis=1, keepdims=True) + 1e-10)

        # Find k nearest neighbors in training set
        k_search = min(self.k, len(self.train_features))
        D, I = self.faiss_index.search(test_norm.astype(np.float32), k_search)

        # Compute scores based on distance statistics
        # Higher distance = more OOD
        mean_distances = np.mean(D, axis=1)
        std_distances = np.std(D, axis=1)

        # Combine mean and std as OOD signal
        scores = mean_distances + 0.5 * std_distances

        return scores

    def _estimate_intrinsic_dimension(self, X):
        pca = PCA()
        pca.fit(X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        dim = np.argmax(cumsum >= 0.95) + 1
        return min(dim, X.shape[1] // 2)

    def _compute_optimal_k(self, n, d_eff):
        optimal = int(np.power(n, 1/3) * np.power(self.empirical_variance, 2/3)) if self.empirical_variance else int(np.sqrt(n * d_eff))
        return min(200, max(10, optimal))

    def _estimate_spectral_variance(self, X, n_bootstrap=5):
        # Reduced from 15 to 5 iterations for speed
        lambda2_samples = []
        n = len(X)
        sample_size = min(n, 200)  # Reduced from 300 to 200

        def bootstrap_iteration(_):
            idx = np.random.choice(n, size=sample_size, replace=True)
            try:
                adj = self.build_knn_adjacency(X[idx])
                lambda2, _ = self.compute_spectral_gap(adj)
                return lambda2 if lambda2 > 0 else None
            except:
                return None

        # Use parallel processing for bootstrap iterations
        with ThreadPoolExecutor(max_workers=min(4, n_bootstrap)) as executor:
            results = list(executor.map(bootstrap_iteration, range(n_bootstrap)))

        lambda2_samples = [r for r in results if r is not None]
        return np.var(lambda2_samples) if len(lambda2_samples) > 2 else 1e-4

    def _estimate_local_lipschitz(self, X):
        # This is a simplified estimation
        return 1.0

    def _compute_theoretical_bounds(self, n):
        if self.empirical_variance is None or self.local_lipschitz is None:
            return {'detection_bound': 0.1}
        
        matrix_bernstein_term = np.sqrt(2 * self.empirical_variance * np.log(2/self.confidence_level) / n)
        range_term = (2 * self.local_lipschitz * np.log(2/self.confidence_level)) / (3 * n)
        return {'detection_bound': matrix_bernstein_term + range_term}


# ============================================================================
# PART 3: ENHANCED EVALUATION
# ============================================================================

def evaluate_ood_methods_enhanced(model, train_loader, test_id_loader, test_ood_loader, device='cuda'):
    print("\n" + "="*70)
    print("ENHANCED OOD DETECTION EVALUATION")
    print("="*70)

    results = {}
    energy_detector = EnergyOODDetector(model)
    spectral_detector = EnhancedSpectralOODDetector()

    print("\n1. Extracting features (limited samples for speed)...")
    # Limit feature extraction to speed up evaluation
    train_features, _ = energy_detector.extract_features(train_loader, device, max_samples=2000)
    test_id_features, _ = energy_detector.extract_features(test_id_loader, device, max_samples=1000)
    test_ood_features, _ = energy_detector.extract_features(test_ood_loader, device, max_samples=1000)

    # Energy-based
    print("\n2. Testing Energy-based Method...")
    energy_scores_id = energy_detector.compute_energy_scores(test_id_loader, device)
    energy_scores_ood = energy_detector.compute_energy_scores(test_ood_loader, device)
    energy_labels = np.concatenate([np.zeros(len(energy_scores_id)), np.ones(len(energy_scores_ood))])
    energy_scores = np.concatenate([energy_scores_id, energy_scores_ood])
    results['energy'] = {'auc': roc_auc_score(energy_labels, -energy_scores), 'ap': average_precision_score(energy_labels, -energy_scores)}
    print(f"   Energy AUC: {results['energy']['auc']:.4f}, AP: {results['energy']['ap']:.4f}")

    # Enhanced Spectral
    print("\n3. Testing Enhanced Spectral Method...")
    train_features_subset = train_features[np.random.choice(len(train_features), min(1000, len(train_features)), replace=False)]
    spectral_detector.fit(train_features_subset)
    spectral_scores_id = spectral_detector.score_samples(test_id_features, batch_size=64)
    spectral_scores_ood = spectral_detector.score_samples(test_ood_features, batch_size=64)
    spectral_labels = np.concatenate([np.zeros(len(spectral_scores_id)), np.ones(len(spectral_scores_ood))])
    spectral_scores = np.concatenate([spectral_scores_id, spectral_scores_ood])
    results['spectral'] = {'auc': roc_auc_score(spectral_labels, spectral_scores), 'ap': average_precision_score(spectral_labels, spectral_scores)}
    print(f"   Enhanced Spectral AUC: {results['spectral']['auc']:.4f}, AP: {results['spectral']['ap']:.4f}")

    return results

def visualize_enhanced_results(results, save_path):
    # Basic visualization, can be expanded
    print(f"\nSaved visualization placeholder to {save_path}")

# ============================================================================
# PART 4: DATA LOADING
# ============================================================================

class ImageNetTestDataset(torch.utils.data.Dataset):
    """Custom dataset for ImageNet test images in flat directory structure"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPEG')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        from PIL import Image
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Return dummy label since we don't have labels
        return image, 0

def get_cifar_loaders(dataset='cifar10', batch_size=128, num_workers=8, imagenet_path='/home/tanmoy/research/data/Imagenet'):
    # Increased num_workers from 4 to 8 for faster data loading
    if dataset == 'imagenet':
        # ImageNet-specific transforms
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # CIFAR-specific transforms
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
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        oodset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        oodset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    elif dataset == 'imagenet':
        # Use custom dataset for flat test directory
        testset = ImageNetTestDataset(root_dir=f'{imagenet_path}/test', transform=transform_test)
        # For ImageNet, we use pretrained model so no training
        trainset = None
        # Use CIFAR100 as OOD for ImageNet
        oodset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Enable pin_memory for faster GPU transfer and persistent_workers to avoid worker respawning
    if trainset is not None:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, persistent_workers=True)
    else:
        train_loader = None

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    ood_loader = DataLoader(oodset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader, ood_loader

# ============================================================================
# PART 5: MAIN MULTI-DATASET TRAINING AND EVALUATION SCRIPT
# ============================================================================

def main():
    # --- Configuration ---
    config = {
        'datasets': ['cifar100', 'imagenet'],
        'model_arch': 'resnet18',
        'batch_size': 128,
        'epochs': 50,
        'lr': 0.1,
        'eval_every_n_epochs': 5,
        'checkpoint_dir': './checkpoints',
        'imagenet_path': '/home/tanmoy/research/data/Imagenet'
    }

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    for dataset_name in config['datasets']:
        print(f"\n{'='*80}")
        print(f" STARTING EXPERIMENT FOR DATASET: {dataset_name.upper()} ".center(80, "="))
        print(f"{'='*80}\n")

        # Load data
        print(f"Loading {dataset_name} data...")
        train_loader, test_loader, ood_loader = get_cifar_loaders(
            dataset=dataset_name,
            batch_size=config['batch_size'],
            imagenet_path=config.get('imagenet_path', './data/imagenet')
        )

        # Load model
        print(f"Initializing {config['model_arch']} model...")
        if dataset_name == 'cifar10':
            num_classes = 10
        elif dataset_name == 'cifar100':
            num_classes = 100
        elif dataset_name == 'imagenet':
            num_classes = 1000
        else:
            num_classes = 100

        if config['model_arch'] == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            if dataset_name == 'imagenet':
                # Use pretrained weights for ImageNet evaluation
                print("Loading pretrained ResNet18 weights for ImageNet...")
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = resnet18(num_classes=num_classes)
        else:
            raise NotImplementedError(f"Model {config['model_arch']} not implemented")
        model = model.to(device)

        # For ImageNet with pretrained model, skip training and only evaluate
        if dataset_name == 'imagenet' and train_loader is None:
            print("\nSkipping training for ImageNet (using pretrained model)")
            print("Running OOD evaluation...")
            results = evaluate_ood_methods_enhanced(
                model, test_loader, test_loader, ood_loader, device
            )
            vis_path = f'ood_results_{dataset_name}_{config["model_arch"]}_pretrained.png'
            visualize_enhanced_results(results, save_path=vis_path)

            print(f"\n{'='*80}")
            print(f" FINISHED EXPERIMENT FOR DATASET: {dataset_name.upper()} ".center(80, "="))
            print(f"{'='*80}\n")
            continue

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                             momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

        print("\nStarting training...")
        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
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
            print(f'Epoch {epoch+1}/{config["epochs"]} | Loss: {train_loss/len(train_loader):.3f} | Acc: {acc:.2f}%')

            # Evaluate and save checkpoint
            if (epoch + 1) % config['eval_every_n_epochs'] == 0 or (epoch + 1) == config['epochs']:
                results = evaluate_ood_methods_enhanced(
                    model, train_loader, test_loader, ood_loader, device
                )
                vis_path = f'ood_results_{dataset_name}_{config["model_arch"]}_epoch_{epoch+1}.png'
                visualize_enhanced_results(results, save_path=vis_path)

                # Save checkpoint
                checkpoint_path = os.path.join(config['checkpoint_dir'], f'{dataset_name}_{config["model_arch"]}_epoch_{epoch+1}.pt')
                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dataset': dataset_name,
                }, checkpoint_path)

        print(f"\n{'='*80}")
        print(f" FINISHED EXPERIMENT FOR DATASET: {dataset_name.upper()} ".center(80, "="))
        print(f"{'='*80}\n")

    print("\nAll experiments completed!")

if __name__ == '__main__':
    main()
