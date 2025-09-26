import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from scipy.sparse import csgraph, csr_matrix
import time
import pickle
from pathlib import Path
import os
import argparse
from sklearn.metrics import roc_auc_score

from make_datasets import *
from models.wrn_ssnd import *
from models.resnet import *
from models.mlp import *

class CachedSpectralOODDetector:
    """
    Efficient spectral OOD detection with feature caching and fast k-NN search
    """
    def __init__(self, k=20, cache_dir='./spectral_cache', use_gpu=True):
        self.k = k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Core cached components
        self.train_features = None
        self.feature_index = None  # FAISS index for fast k-NN
        self.baseline_lambda2 = None
        self.baseline_graph = None
        self.graph_laplacian = None

    def _get_cache_path(self, name):
        """Get path for cached file"""
        return self.cache_dir / f"{name}.pkl"

    def _cache_exists(self, name):
        """Check if cache file exists"""
        return self._get_cache_path(name).exists()

    def _save_cache(self, data, name):
        """Save data to cache"""
        with open(self._get_cache_path(name), 'wb') as f:
            pickle.dump(data, f)

    def _load_cache(self, name):
        """Load data from cache"""
        with open(self._get_cache_path(name), 'rb') as f:
            return pickle.load(f)

    def extract_and_cache_features(self, model, dataloader, cache_name='train_features'):
        """Extract features and cache them"""
        cache_path = self._get_cache_path(cache_name)

        if self._cache_exists(cache_name):
            print(f"Loading cached features from {cache_path}")
            features, labels = self._load_cache(cache_name)
        else:
            print(f"Extracting features (will be cached to {cache_path})")
            features, labels = self._extract_features(model, dataloader)
            self._save_cache((features, labels), cache_name)

        return features, labels

    def _extract_features(self, model, dataloader):
        """Extract features from model"""
        model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            for data, labels in dataloader:
                if self.use_gpu:
                    data = data.cuda()

                # Get features before final classifier
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(data)
                else:
                    # Hook into penultimate layer
                    features = self._get_penultimate_features(model, data)

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())

        return np.vstack(features_list), np.concatenate(labels_list)

    def _get_penultimate_features(self, model, data):
        """Extract penultimate layer features"""
        features = []
        def hook(module, input, output):
            features.append(input[0] if isinstance(input, tuple) else input)

        # Register hook on final linear layer
        if hasattr(model, 'fc'):
            handle = model.fc.register_forward_hook(hook)
        elif hasattr(model, 'linear'):
            handle = model.linear.register_forward_hook(hook)
        else:
            raise ValueError("Cannot find final layer")

        _ = model(data)
        handle.remove()

        return features[0]

    def build_cached_index(self, features, index_name='faiss_index'):
        """Build FAISS index for fast k-NN search"""
        if self._cache_exists(index_name):
            print(f"Loading cached FAISS index")
            index = faiss.read_index(str(self._get_cache_path(index_name)))
            if self.use_gpu:
                # Move to GPU
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            print(f"Building FAISS index (will be cached)")
            # Normalize features
            features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

            # Build index
            d = features_norm.shape[1]
            if self.use_gpu:
                # GPU index
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, d)  # Inner product for cosine similarity
            else:
                index = faiss.IndexFlatIP(d)

            index.add(features_norm.astype(np.float32))

            # Save CPU version
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, str(self._get_cache_path(index_name)))
            else:
                faiss.write_index(index, str(self._get_cache_path(index_name)))

        return index

    def compute_baseline_spectrum(self, features, cache_name='baseline_spectrum'):
        """Compute and cache baseline spectral properties"""
        if self._cache_exists(cache_name):
            print("Loading cached baseline spectrum")
            cached_data = self._load_cache(cache_name)
            self.baseline_lambda2 = cached_data['lambda2']
            self.baseline_eigenvalues = cached_data['eigenvalues']
        else:
            print("Computing baseline spectrum (will be cached)")
            # Build sparse k-NN graph
            sparse_graph = self._build_sparse_graph(features)

            # Compute Laplacian eigenvalues
            laplacian = csgraph.laplacian(sparse_graph, normed=True)

            # Compute only first few eigenvalues for efficiency
            from scipy.sparse.linalg import eigsh
            eigenvalues, _ = eigsh(laplacian, k=min(10, len(features)-1),
                                  sigma=0, which='LM')
            eigenvalues = np.sort(eigenvalues)

            self.baseline_lambda2 = eigenvalues[1]
            self.baseline_eigenvalues = eigenvalues

            # Cache results
            self._save_cache({
                'lambda2': self.baseline_lambda2,
                'eigenvalues': eigenvalues
            }, cache_name)

        print(f"Baseline λ2: {self.baseline_lambda2:.6f}")

    def _build_sparse_graph(self, features):
        """Build sparse k-NN graph efficiently"""
        n = len(features)
        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

        # Use FAISS for fast k-NN search
        index = faiss.IndexFlatIP(features_norm.shape[1])
        index.add(features_norm.astype(np.float32))

        # Find k-NN for all points
        similarities, indices = index.search(features_norm.astype(np.float32), self.k + 1)

        # Build sparse matrix
        row_indices = []
        col_indices = []
        data = []

        for i in range(n):
            # Skip self-connection (first neighbor)
            for j in range(1, self.k + 1):
                if indices[i, j] < n:  # Valid neighbor
                    row_indices.append(i)
                    col_indices.append(indices[i, j])
                    data.append(similarities[i, j])

        # Create sparse matrix
        sparse_graph = csr_matrix((data, (row_indices, col_indices)),
                                 shape=(n, n))

        # Symmetrize
        sparse_graph = (sparse_graph + sparse_graph.T) / 2

        return sparse_graph

    def fit(self, model, train_loader):
        """Fit the detector with caching"""
        print("="*50)
        print("Fitting Spectral OOD Detector with Caching")
        print("="*50)

        # 1. Extract and cache features
        train_features, train_labels = self.extract_and_cache_features(
            model, train_loader, 'train_features'
        )
        self.train_features = train_features

        # 2. Build and cache FAISS index
        self.feature_index = self.build_cached_index(train_features)

        # 3. Compute and cache baseline spectrum
        self.compute_baseline_spectrum(train_features)

        print("Fitting complete!")

    def score_online(self, model, test_loader, batch_size=100):
        """Score test samples online (no caching)"""
        model.eval()
        scores = []

        with torch.no_grad():
            for data, _ in test_loader:
                if self.use_gpu:
                    data = data.cuda()

                # Extract features
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(data)
                else:
                    features = self._get_penultimate_features(model, data)

                features = features.cpu().numpy()

                # Score batch
                batch_scores = self._score_batch_fast(features)
                scores.extend(batch_scores)

        return np.array(scores)

    def _score_batch_fast(self, test_features):
        """Fast scoring using incremental updates"""
        test_features_norm = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
        scores = []

        for i in range(len(test_features)):
            # Method 1: Local perturbation analysis (fast approximation)
            score = self._local_perturbation_score(test_features_norm[i])
            scores.append(score)

        return scores

    def _local_perturbation_score(self, test_feature):
        """Fast approximation based on local graph perturbation"""
        test_feature = test_feature.reshape(1, -1).astype(np.float32)

        # Find k nearest neighbors in training set
        similarities, indices = self.feature_index.search(test_feature, self.k)

        # Compute connectivity score
        # Higher connectivity to ID samples = lower OOD score
        connectivity = np.mean(similarities[0])

        # Estimate spectral change (approximation)
        # Based on how well connected the test point is
        estimated_spectral_change = (1 - connectivity) * self.baseline_lambda2

        return estimated_spectral_change

class FastSpectralComparison:
    """Fast comparison framework with caching"""

    def __init__(self, cache_dir='./spectral_cache'):
        self.cache_dir = cache_dir
        self.detector = CachedSpectralOODDetector(cache_dir=cache_dir)

    def compare_with_energy(self, model, train_loader, test_loader_id, test_loader_ood):
        """Compare cached spectral with Wang & Li's energy method"""

        # 1. Fit spectral detector (with caching)
        start_time = time.time()
        self.detector.fit(model, train_loader)
        fit_time = time.time() - start_time
        print(f"Spectral fitting time: {fit_time:.2f}s")

        # 2. Score test samples
        print("\nScoring test samples...")
        start_time = time.time()
        spectral_scores_id = self.detector.score_online(model, test_loader_id)
        spectral_scores_ood = self.detector.score_online(model, test_loader_ood)
        spectral_score_time = time.time() - start_time

        # 3. Compute energy scores for comparison
        print("\nComputing energy scores...")
        start_time = time.time()
        energy_scores_id = self._compute_energy_scores(model, test_loader_id)
        energy_scores_ood = self._compute_energy_scores(model, test_loader_ood)
        energy_score_time = time.time() - start_time

        # 4. Evaluate
        spectral_labels = np.concatenate([np.zeros(len(spectral_scores_id)),
                                         np.ones(len(spectral_scores_ood))])
        spectral_all = np.concatenate([spectral_scores_id, spectral_scores_ood])
        spectral_auc = roc_auc_score(spectral_labels, spectral_all)

        energy_labels = np.concatenate([np.zeros(len(energy_scores_id)),
                                       np.ones(len(energy_scores_ood))])
        energy_all = np.concatenate([energy_scores_id, energy_scores_ood])
        energy_auc = roc_auc_score(energy_labels, -energy_all)  # Negative for energy

        print("\n" + "="*50)
        print("RESULTS WITH CACHING")
        print("="*50)
        print(f"Spectral AUC: {spectral_auc:.4f} (time: {spectral_score_time:.2f}s)")
        print(f"Energy AUC: {energy_auc:.4f} (time: {energy_score_time:.2f}s)")
        print(f"Speedup from caching: {energy_score_time/spectral_score_time:.2f}x")
        print("="*50)

        return {
            'spectral_auc': spectral_auc,
            'energy_auc': energy_auc,
            'spectral_time': spectral_score_time,
            'energy_time': energy_score_time
        }

    def _compute_energy_scores(self, model, dataloader, T=1.0):
        """Compute energy scores"""
        model.eval()
        scores = []

        with torch.no_grad():
            for data, _ in dataloader:
                if torch.cuda.is_available():
                    data = data.cuda()

                logits = model(data)
                energy = -T * torch.logsumexp(logits / T, dim=1)
                scores.extend(energy.cpu().numpy())

        return np.array(scores)

def extract_all_features(args, model, cache_dir='./spectral_features'):
    """Extract and cache features from all datasets"""

    # Create cache directory with dataset name
    cache_dir = Path(cache_dir) / args.dataset
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Create detector
    detector = CachedSpectralOODDetector(cache_dir=cache_dir)

    print("="*60)
    print("EXTRACTING FEATURES FROM ALL DATASETS")
    print("="*60)

    # Make datasets
    print("\nLoading datasets...")
    state = {k: v for k, v in args._get_kwargs()}
    state['seed'] = 1  # Add default seed
    state['batch_size'] = args.batch_size
    state['test_bs'] = 200
    state['oe_batch_size'] = 256
    train_loader_in, train_loader_in_noaug, train_loader_aux_in, train_loader_aux_in_cor, \
    train_loader_aux_out, test_loader_in, test_loader_cor, test_loader_ood, \
    valid_loader_in, valid_loader_aux, valid_loader_aux_in, valid_loader_aux_cor, \
    valid_loader_aux_out = make_datasets(
        args.dataset, args.aux_out_dataset, args.test_out_dataset,
        state, args.alpha, args.pi_1, args.pi_2, args.cortype
    )

    all_features = {}

    # 1. Train in-distribution features
    print(f"\n1. Extracting {args.dataset} train in-distribution features...")
    train_in_features, train_in_labels = detector.extract_and_cache_features(
        model, train_loader_in, f'{args.dataset}_train_in_features'
    )
    all_features['train_in'] = {'features': train_in_features, 'labels': train_in_labels}
    print(f"   Shape: {train_in_features.shape}")

    # 2. Train auxiliary in-distribution features
    print(f"\n2. Extracting {args.dataset} train auxiliary in-distribution features...")
    train_aux_in_features, train_aux_in_labels = detector.extract_and_cache_features(
        model, train_loader_aux_in, f'{args.dataset}_train_aux_in_features'
    )
    all_features['train_aux_in'] = {'features': train_aux_in_features, 'labels': train_aux_in_labels}
    print(f"   Shape: {train_aux_in_features.shape}")

    # 3. Train auxiliary corrupted features
    print(f"\n3. Extracting {args.dataset} train auxiliary corrupted ({args.cortype}) features...")
    train_aux_cor_features, train_aux_cor_labels = detector.extract_and_cache_features(
        model, train_loader_aux_in_cor, f'{args.dataset}_train_aux_cor_{args.cortype}_features'
    )
    all_features['train_aux_cor'] = {'features': train_aux_cor_features, 'labels': train_aux_cor_labels}
    print(f"   Shape: {train_aux_cor_features.shape}")

    # 4. Train auxiliary OOD features
    print(f"\n4. Extracting {args.dataset} train auxiliary OOD ({args.aux_out_dataset}) features...")
    train_aux_out_features, train_aux_out_labels = detector.extract_and_cache_features(
        model, train_loader_aux_out, f'{args.dataset}_train_aux_out_{args.aux_out_dataset}_features'
    )
    all_features['train_aux_out'] = {'features': train_aux_out_features, 'labels': train_aux_out_labels}
    print(f"   Shape: {train_aux_out_features.shape}")

    # 5. Test in-distribution features
    print(f"\n5. Extracting {args.dataset} test in-distribution features...")
    test_in_features, test_in_labels = detector.extract_and_cache_features(
        model, test_loader_in, f'{args.dataset}_test_in_features'
    )
    all_features['test_in'] = {'features': test_in_features, 'labels': test_in_labels}
    print(f"   Shape: {test_in_features.shape}")

    # 6. Test corrupted features
    print(f"\n6. Extracting {args.dataset} test corrupted ({args.cortype}) features...")
    test_cor_features, test_cor_labels = detector.extract_and_cache_features(
        model, test_loader_cor, f'{args.dataset}_test_cor_{args.cortype}_features'
    )
    all_features['test_cor'] = {'features': test_cor_features, 'labels': test_cor_labels}
    print(f"   Shape: {test_cor_features.shape}")

    # 7. Test OOD features
    print(f"\n7. Extracting {args.dataset} test OOD ({args.test_out_dataset}) features...")
    test_ood_features, test_ood_labels = detector.extract_and_cache_features(
        model, test_loader_ood, f'{args.dataset}_test_ood_{args.test_out_dataset}_features'
    )
    all_features['test_ood'] = {'features': test_ood_features, 'labels': test_ood_labels}
    print(f"   Shape: {test_ood_features.shape}")

    # 8. Valid in-distribution features
    print(f"\n8. Extracting {args.dataset} valid in-distribution features...")
    valid_in_features, valid_in_labels = detector.extract_and_cache_features(
        model, valid_loader_in, f'{args.dataset}_valid_in_features'
    )
    all_features['valid_in'] = {'features': valid_in_features, 'labels': valid_in_labels}
    print(f"   Shape: {valid_in_features.shape}")

    # 9. Valid auxiliary features
    print(f"\n9. Extracting {args.dataset} valid auxiliary features...")
    valid_aux_features, valid_aux_labels = detector.extract_and_cache_features(
        model, valid_loader_aux, f'{args.dataset}_valid_aux_features'
    )
    all_features['valid_aux'] = {'features': valid_aux_features, 'labels': valid_aux_labels}
    print(f"   Shape: {valid_aux_features.shape}")

    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print("\nSummary of extracted features:")
    for name, data in all_features.items():
        print(f"  {name:20s}: {data['features'].shape}")

    # Save all features summary with dataset name
    summary_path = cache_dir / f'{args.dataset}_all_features_summary.pkl'
    with open(summary_path, 'wb') as f:
        summary_data = {
            'dataset': args.dataset,
            'aux_out_dataset': args.aux_out_dataset,
            'test_out_dataset': args.test_out_dataset,
            'cortype': args.cortype,
            'feature_shapes': {name: data['features'].shape for name, data in all_features.items()}
        }
        pickle.dump(summary_data, f)
    print(f"\nFeature summary saved to: {summary_path}")

    return all_features, detector

def main():
    parser = argparse.ArgumentParser(description='Extract features using Spectral OOD Detector')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'MNIST', 'imagenet100'],
                       help='Choose dataset')
    parser.add_argument('--aux_out_dataset', type=str, default='FashionMNIST',
                       help='Auxiliary out of distribution dataset')
    parser.add_argument('--test_out_dataset', type=str, default='FashionMNIST',
                       help='Test out of distribution dataset')

    # Model arguments
    parser.add_argument('--model', type=str, default='wrn',
                       choices=['allconv', 'wrn', 'densenet', 'mlp'],
                       help='Choose architecture')
    parser.add_argument('--layers', default=40, type=int,
                       help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float,
                       help='dropout probability')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='number of labeled samples')
    parser.add_argument('--pi_1', type=float, default=0.5,
                       help='proportion of ood data in auxiliary dataset')
    parser.add_argument('--pi_2', type=float, default=0.5,
                       help='proportion of ood data in auxiliary dataset')
    parser.add_argument('--cortype', type=str, default='gaussian_noise',
                       help='corrupted type of images')

    # Model checkpoint
    parser.add_argument('--load_pretrained', type=str, required=True,
                       help='Path to pretrained model')

    # Cache directory
    parser.add_argument('--cache_dir', type=str, default='./spectral_features',
                       help='Directory to cache features')

    # GPU
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use')

    args = parser.parse_args()

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    # Determine number of classes
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'MNIST':
        num_classes = 10
    else:
        num_classes = 100

    # Load model
    print(f"\nLoading model from: {args.load_pretrained}")
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate, args=args)

    if os.path.isfile(args.load_pretrained):
        # Load model with appropriate device mapping
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(args.load_pretrained))
        else:
            net.load_state_dict(torch.load(args.load_pretrained, map_location=torch.device('cpu')))
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Could not find model: {args.load_pretrained}")

    if torch.cuda.is_available():
        net = net.cuda()

    # Extract features from all datasets
    all_features, detector = extract_all_features(args, net, cache_dir=args.cache_dir)

    # Run spectral analysis on train features
    print("\n" + "="*60)
    print("RUNNING SPECTRAL ANALYSIS")
    print("="*60)

    # Build k-NN graph and compute spectrum for training data
    detector.compute_baseline_spectrum(all_features['train_in']['features'])

    # Compare with test sets
    print("\nComputing OOD detection performance...")

    # Spectral scores
    test_in_scores = detector._score_batch_fast(all_features['test_in']['features'])
    test_ood_scores = detector._score_batch_fast(all_features['test_ood']['features'])
    test_cor_scores = detector._score_batch_fast(all_features['test_cor']['features'])

    # Compute AUC
    from sklearn.metrics import roc_auc_score

    # Test ID vs Test OOD
    labels = np.concatenate([np.zeros(len(test_in_scores)), np.ones(len(test_ood_scores))])
    scores = np.concatenate([test_in_scores, test_ood_scores])
    auc_ood = roc_auc_score(labels, scores)

    # Test ID vs Test Corrupted
    labels_cor = np.concatenate([np.zeros(len(test_in_scores)), np.ones(len(test_cor_scores))])
    scores_cor = np.concatenate([test_in_scores, test_cor_scores])
    auc_cor = roc_auc_score(labels_cor, scores_cor)

    print(f"\nSpectral OOD Detection Results:")
    print(f"  Test ID vs Test OOD AUC: {auc_ood:.4f}")
    print(f"  Test ID vs Test Corrupted AUC: {auc_cor:.4f}")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()