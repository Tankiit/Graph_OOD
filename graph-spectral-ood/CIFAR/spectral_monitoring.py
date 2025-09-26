"""
Spectral monitoring functions for Graph OOD detection
Direct integration with your existing code structure
"""

import numpy as np
import torch
import faiss
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time

def compute_spectral_gap(features, k=50, num_eigenvalues=10, return_full=False):
    """
    Compute spectral gap from features using k-NN graph

    Args:
        features: torch.Tensor or numpy array of shape (n_samples, n_features)
        k: number of nearest neighbors (matching your knn_k)
        num_eigenvalues: number of smallest eigenvalues to compute
        return_full: if True, return eigenvalues and eigenvectors

    Returns:
        gap: spectral gap (λ_2 - λ_1)
        (optional) eigenvalues, eigenvectors if return_full=True
    """
    # Convert to numpy if needed
    if torch.is_tensor(features):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = features

    n_samples = features_np.shape[0]

    # Handle small sample case
    if n_samples <= k:
        return 0.0 if not return_full else (0.0, None, None)

    # Build k-NN graph using FAISS (same as your knn implementation)
    index = faiss.IndexFlatL2(features_np.shape[1])
    index.add(features_np.astype(np.float32))

    # Search for k+1 neighbors (including self)
    D, I = index.search(features_np.astype(np.float32), k + 1)

    # Remove self-connections
    D = D[:, 1:]
    I = I[:, 1:]

    # Compute Gaussian weights (adaptive bandwidth)
    sigma = np.median(D)
    if sigma == 0:
        sigma = 1.0
    weights = np.exp(-D**2 / (2 * sigma**2))

    # Build sparse adjacency matrix
    row_idx = np.repeat(np.arange(n_samples), k)
    col_idx = I.flatten()
    data = weights.flatten()

    # Create sparse matrix
    A = csr_matrix((data, (row_idx, col_idx)), shape=(n_samples, n_samples))
    A = 0.5 * (A + A.T)  # Symmetrize

    # Compute degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1e-10  # Avoid division by zero

    # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    D_sqrt_inv = 1.0 / np.sqrt(degrees)
    D_sqrt_inv_sparse = csr_matrix((D_sqrt_inv, (np.arange(n_samples), np.arange(n_samples))),
                                  shape=(n_samples, n_samples))
    L = csr_matrix(np.eye(n_samples)) - D_sqrt_inv_sparse @ A @ D_sqrt_inv_sparse

    # Compute eigenvalues
    try:
        num_eigs = min(num_eigenvalues, n_samples - 1)
        eigenvalues, eigenvectors = eigsh(L, k=num_eigs, which='SM', tol=1e-6)

        # Sort eigenvalues
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Compute spectral gap
        gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0

        if return_full:
            return gap, eigenvalues, eigenvectors
        else:
            return gap

    except Exception as e:
        print(f"Warning: Eigenvalue computation failed: {e}")
        return 0.0 if not return_full else (0.0, None, None)


def monitor_spectral_gap_wild(in_features, wild_features, pi_1, pi_2, k=50):
    """
    Monitor how wild data affects spectral gap
    Specifically designed for your pi_1, pi_2 framework

    Args:
        in_features: in-distribution features
        wild_features: wild (auxiliary) features
        pi_1: mixture proportion for covariate shift
        pi_2: mixture proportion for semantic shift
        k: number of nearest neighbors

    Returns:
        dict with gap measurements
    """
    results = {}

    # 1. Pure in-distribution gap
    gap_pure = compute_spectral_gap(in_features, k=k)
    results['gap_pure_in'] = gap_pure

    # 2. Estimate covariate vs semantic components in wild data
    # This is an approximation since we don't have perfect labels
    n_wild = wild_features.shape[0]
    n_cov_estimate = int(n_wild * pi_1 / (pi_1 + pi_2)) if (pi_1 + pi_2) > 0 else 0
    n_sem_estimate = n_wild - n_cov_estimate

    # 3. Mix with different proportions
    if n_wild > 0:
        # Random sampling to simulate mixture
        if torch.is_tensor(wild_features):
            wild_features_np = wild_features.detach().cpu().numpy()
            in_features_np = in_features.detach().cpu().numpy()
        else:
            wild_features_np = wild_features
            in_features_np = in_features

        # Full mixture gap
        features_mixed = np.vstack([in_features_np, wild_features_np])
        gap_mixed = compute_spectral_gap(features_mixed, k=k)
        results['gap_mixed'] = gap_mixed

        # Gap degradation
        results['degradation'] = 1.0 - gap_mixed / gap_pure if gap_pure > 0 else 0.0

        # Theoretical quantities
        results['effective_pi1'] = pi_1
        results['effective_pi2'] = pi_2
        results['wild_ratio'] = n_wild / in_features.shape[0]
    else:
        results['gap_mixed'] = gap_pure
        results['degradation'] = 0.0
        results['wild_ratio'] = 0.0

    return results


def spectral_monitoring_epoch(net, train_loader_in, train_loader_aux_in,
                            train_loader_aux_in_cor, train_loader_aux_out,
                            args, epoch, sample_batches=5):
    """
    Spectral monitoring to integrate into your training loop
    Samples a few batches to compute spectral statistics

    Args:
        net: your model
        train_loaders: your data loaders
        args: your args including pi_1, pi_2
        epoch: current epoch
        sample_batches: number of batches to sample for monitoring

    Returns:
        spectral_stats: dictionary of spectral metrics
    """
    net.eval()

    # Storage for features
    in_features_list = []
    wild_features_list = []

    # Sample a few batches
    loaders = zip(train_loader_in, train_loader_aux_in,
                 train_loader_aux_in_cor, train_loader_aux_out)

    with torch.no_grad():
        for batch_idx, (in_set, aux_in_set, aux_in_cor_set, aux_out_set) in enumerate(loaders):
            if batch_idx >= sample_batches:
                break

            # Get in-distribution features
            data_in = in_set[0].cuda()

            # Get features based on your network architecture
            if hasattr(net, 'forward_backbone'):
                _, features_in = net.forward_backbone(data_in, return_feat=True)
            else:
                # Fallback to forward pass and extract features
                # Modify based on your network architecture
                output = net(data_in)
                features_in = net.get_features(data_in) if hasattr(net, 'get_features') else output

            in_features_list.append(features_in)

            # Create wild mixture (using your mix_batches logic)
            # Import mix_batches function
            from train import mix_batches
            aux_set = mix_batches(aux_in_set, aux_in_cor_set, aux_out_set)
            data_wild = aux_set.cuda()

            if hasattr(net, 'forward_backbone'):
                _, features_wild = net.forward_backbone(data_wild, return_feat=True)
            else:
                output = net(data_wild)
                features_wild = net.get_features(data_wild) if hasattr(net, 'get_features') else output

            wild_features_list.append(features_wild)

    # Concatenate features
    in_features = torch.cat(in_features_list, dim=0)
    wild_features = torch.cat(wild_features_list, dim=0) if wild_features_list else torch.empty(0)

    # Compute spectral gaps
    spectral_stats = monitor_spectral_gap_wild(
        in_features, wild_features, args.pi_1, args.pi_2, k=50
    )

    # Add epoch info
    spectral_stats['epoch'] = epoch

    # Print summary
    print(f"\n[Spectral Monitor] Epoch {epoch}:")
    print(f"  Pure IN gap: {spectral_stats['gap_pure_in']:.4f}")
    print(f"  Mixed gap: {spectral_stats.get('gap_mixed', 0):.4f}")
    print(f"  Degradation: {spectral_stats.get('degradation', 0):.2%}")
    print(f"  Wild ratio: {spectral_stats.get('wild_ratio', 0):.2f}")

    net.train()
    return spectral_stats


def adaptive_mixture_adjustment(spectral_stats, args, threshold=0.3):
    """
    Adaptively adjust pi_1 and pi_2 based on spectral feedback

    Args:
        spectral_stats: output from spectral monitoring
        args: your args object with pi_1, pi_2
        threshold: maximum acceptable degradation

    Returns:
        adjusted pi_1, pi_2
    """
    degradation = spectral_stats.get('degradation', 0)

    if degradation > threshold:
        # Too much spectral degradation - adjust mixture
        print(f"\n[Adaptive Adjustment] High degradation {degradation:.2%}, adjusting mixture...")

        # Reduce harmful semantic shift
        new_pi_2 = args.pi_2 * 0.9

        # Increase helpful covariate shift (with upper bound)
        new_pi_1 = min(args.pi_1 * 1.1, 0.5)

        print(f"  π₁: {args.pi_1:.3f} → {new_pi_1:.3f}")
        print(f"  π₂: {args.pi_2:.3f} → {new_pi_2:.3f}")

        return new_pi_1, new_pi_2

    return args.pi_1, args.pi_2


def spectral_regularization_loss(features, k=50, alpha=0.01):
    """
    Add spectral gap as regularization to maintain good structure
    """
    gap = compute_spectral_gap(features, k=k)
    # Maximize gap (minimize negative gap)
    device = features.device if torch.is_tensor(features) else torch.device('cpu')
    loss_spectral = -alpha * torch.log(torch.tensor(gap + 1e-6).to(device))
    return loss_spectral


def spectral_constraint_term(features_in, features_out, k=50):
    """
    Constraint: gap(in) - gap(out) > margin
    """
    gap_in = compute_spectral_gap(features_in, k=k)
    gap_out = compute_spectral_gap(features_out, k=k)

    margin = 0.1  # minimum gap difference
    constraint = gap_in - gap_out - margin

    device = features_in.device if torch.is_tensor(features_in) else torch.device('cpu')
    return torch.tensor(constraint).to(device)


def test_spectral_monitoring():
    """Test function to verify spectral monitoring works"""
    # Create dummy data
    n_samples = 200
    n_features = 128

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # In-distribution: clustered
    features_in = torch.randn(n_samples, n_features).to(device)
    features_in[:100] += 2.0  # Create two clusters

    # Wild data: mixed
    features_wild = torch.randn(n_samples, n_features).to(device)
    features_wild[::2] += 1.5  # Some structure

    # Test gap computation
    gap_in = compute_spectral_gap(features_in, k=20)
    print(f"In-distribution gap: {gap_in:.4f}")

    # Test monitoring
    results = monitor_spectral_gap_wild(
        features_in, features_wild,
        pi_1=0.3, pi_2=0.2, k=20
    )

    print(f"Monitoring results: {results}")

    return results