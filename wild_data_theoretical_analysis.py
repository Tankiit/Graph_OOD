"""
Wild Data Theoretical Analysis Implementation
Validates spectral theory using controlled mixture experiments

This implements the theoretical framework from the integrated wild data section
to empirically validate spectral mixture decomposition on your datasets.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class WildDataConfig:
    """Configuration for wild data experiments"""
    pi_covariate_range: List[float] = None  # [0.1, 0.2, 0.3, 0.4, 0.5]
    pi_semantic_range: List[float] = None   # [0.1, 0.2, 0.3]
    k_neighbors: int = 50
    n_trials: int = 5  # For statistical significance
    confidence_level: float = 0.95

    def __post_init__(self):
        if self.pi_covariate_range is None:
            self.pi_covariate_range = [0.1, 0.2, 0.3, 0.4, 0.5]
        if self.pi_semantic_range is None:
            self.pi_semantic_range = [0.1, 0.2, 0.3]


class WildDataTheoreticalAnalyzer:
    """
    Implements theoretical analysis of wild data mixtures using spectral properties

    Key theoretical components:
    1. Controlled mixture creation following Wang & Li framework
    2. Spectral decomposition analysis
    3. Mixture coefficient estimation
    4. Theoretical bound validation
    """

    def __init__(self, config: WildDataConfig):
        self.config = config
        self.baseline_lambda2 = None
        self.delta_covariate = None
        self.delta_semantic = None

        # Storage for experimental results
        self.results = {
            'mixture_experiments': [],
            'spectral_decomposition': [],
            'estimation_accuracy': [],
            'theoretical_bounds': []
        }

    def extract_features_from_dataloader(self, model, dataloader, device='cuda'):
        """Extract features using your existing model"""
        model.eval()
        features = []
        labels = []

        # Check if model is a simple feature extractor (for synthetic data)
        is_synthetic = hasattr(model, '__class__') and model.__class__.__name__ == 'SimpleFeatureExtractor'

        if is_synthetic:
            # Direct feature extraction for synthetic data
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Extracting features"):
                    if isinstance(batch, (list, tuple)):
                        data, batch_labels = batch[0], batch[1]
                    else:
                        data, batch_labels = batch, None

                    data = data.to(device)
                    feat = data.cpu().numpy()  # Already features
                    features.append(feat)
                    if batch_labels is not None:
                        labels.extend(batch_labels.cpu().numpy())
        else:
            # Hook to capture penultimate layer features for real models
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = input[0] if isinstance(input, tuple) else input
                return hook

            # Register hook (adapt to your model architecture)
            if hasattr(model, 'fc'):
                handle = model.fc.register_forward_hook(get_activation('features'))
            elif hasattr(model, 'linear'):
                handle = model.linear.register_forward_hook(get_activation('features'))
            else:
                raise ValueError("Cannot find linear layer to hook")

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Extracting features"):
                    if isinstance(batch, (list, tuple)):
                        data, batch_labels = batch[0], batch[1]
                    else:
                        data, batch_labels = batch, None

                    data = data.to(device)
                    _ = model(data)

                    feat = activation['features'].cpu().numpy()
                    features.append(feat)
                    if batch_labels is not None:
                        labels.extend(batch_labels.cpu().numpy())

            handle.remove()

        features = np.vstack(features)
        labels = np.array(labels) if labels else None

        return features, labels

    def compute_graph_spectrum(self, features: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Compute graph spectrum from features
        Returns eigenvalues and algebraic connectivity (λ₂)
        """
        if k is None:
            k = self.config.k_neighbors

        n_samples = features.shape[0]
        if n_samples <= k:
            k = n_samples - 1

        # Normalize features for cosine similarity
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(features_norm)
        distances, indices = nbrs.kneighbors(features_norm)

        # Remove self-connections
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Convert to similarities and build adjacency matrix
        similarities = 1 - distances  # cosine distance to cosine similarity
        adjacency = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            adjacency[i, indices[i]] = similarities[i]

        # Symmetrize
        adjacency = (adjacency + adjacency.T) / 2

        # Compute normalized Laplacian and eigenvalues
        try:
            laplacian = csgraph.laplacian(adjacency, normed=True)

            if n_samples > 100:
                # Use sparse solver for efficiency
                eigenvals = eigsh(laplacian, k=min(10, n_samples-2), which='SM', return_eigenvectors=False)
            else:
                eigenvals = np.linalg.eigvalsh(laplacian)

            eigenvals = np.sort(eigenvals)
            lambda2 = eigenvals[1] if len(eigenvals) > 1 else 0.0

            return eigenvals, lambda2

        except Exception as e:
            logging.warning(f"Eigenvalue computation failed: {e}")
            return np.zeros(10), 0.0

    def create_controlled_mixture(self,
                                id_features: np.ndarray,
                                covariate_features: np.ndarray,
                                semantic_features: np.ndarray,
                                pi_c: float,
                                pi_s: float) -> np.ndarray:
        """
        Create controlled wild data mixture following Wang & Li Definition 2.1

        P_wild = (1 - π_c - π_s)P_in + π_c P_covariate + π_s P_semantic
        """
        pi_in = 1 - pi_c - pi_s
        assert pi_in >= 0, f"Invalid mixture coefficients: π_c={pi_c}, π_s={pi_s}"

        # Sample sizes for each component
        total_samples = len(id_features)
        n_in = int(pi_in * total_samples)
        n_cov = int(pi_c * total_samples)
        n_sem = int(pi_s * total_samples)

        # Ensure we have enough samples
        n_in = min(n_in, len(id_features))
        n_cov = min(n_cov, len(covariate_features))
        n_sem = min(n_sem, len(semantic_features))

        # Sample from each distribution
        if n_in > 0:
            id_idx = np.random.choice(len(id_features), n_in, replace=False)
            id_sample = id_features[id_idx]
        else:
            id_sample = np.empty((0, id_features.shape[1]))

        if n_cov > 0:
            cov_idx = np.random.choice(len(covariate_features), n_cov, replace=False)
            cov_sample = covariate_features[cov_idx]
        else:
            cov_sample = np.empty((0, covariate_features.shape[1]))

        if n_sem > 0:
            sem_idx = np.random.choice(len(semantic_features), n_sem, replace=False)
            sem_sample = semantic_features[sem_idx]
        else:
            sem_sample = np.empty((0, semantic_features.shape[1]))

        # Combine samples
        mixed_features = np.vstack([id_sample, cov_sample, sem_sample])

        return mixed_features

    def estimate_mixture_coefficients(self,
                                    lambda2_wild: float,
                                    lambda2_baseline: float) -> Tuple[float, float]:
        """
        Estimate mixture coefficients from spectral shift

        Uses: λ₂(wild) = λ₂(in) + π_c * Δ_c + π_s * Δ_s
        """
        if self.delta_covariate is None or self.delta_semantic is None:
            return 0.0, 0.0

        spectral_shift = lambda2_wild - lambda2_baseline

        # Simple estimation assuming equal contributions
        # In practice, this would require solving a linear system with constraints
        if abs(self.delta_semantic - self.delta_covariate) < 1e-6:
            return 0.0, 0.0

        # Assuming π_c and π_s are small, approximate solution
        total_shift_magnitude = abs(spectral_shift)
        if total_shift_magnitude < 1e-6:
            return 0.0, 0.0

        # Rough estimation - in practice you'd use optimization
        estimated_pi_s = max(0, min(0.5, spectral_shift / self.delta_semantic))
        estimated_pi_c = max(0, min(0.5, -spectral_shift / self.delta_covariate))

        return estimated_pi_c, estimated_pi_s

    def run_mixture_experiments(self,
                              model,
                              id_dataloader,
                              covariate_dataloader,
                              semantic_dataloader,
                              device='cuda') -> Dict:
        """
        Run controlled mixture experiments to validate theoretical predictions
        """
        logging.info("Starting mixture experiments...")

        # Extract features
        logging.info("Extracting ID features...")
        id_features, _ = self.extract_features_from_dataloader(model, id_dataloader, device)

        logging.info("Extracting covariate features...")
        cov_features, _ = self.extract_features_from_dataloader(model, covariate_dataloader, device)

        logging.info("Extracting semantic features...")
        sem_features, _ = self.extract_features_from_dataloader(model, semantic_dataloader, device)

        # Compute baseline spectrum
        _, self.baseline_lambda2 = self.compute_graph_spectrum(id_features)
        logging.info(f"Baseline λ₂: {self.baseline_lambda2:.6f}")

        # Estimate Δ_covariate and Δ_semantic
        _, lambda2_cov = self.compute_graph_spectrum(np.vstack([id_features, cov_features[:len(id_features)//2]]))
        _, lambda2_sem = self.compute_graph_spectrum(np.vstack([id_features, sem_features[:len(id_features)//2]]))

        self.delta_covariate = lambda2_cov - self.baseline_lambda2
        self.delta_semantic = lambda2_sem - self.baseline_lambda2

        logging.info(f"Estimated Δ_covariate: {self.delta_covariate:.6f}")
        logging.info(f"Estimated Δ_semantic: {self.delta_semantic:.6f}")

        # Run mixture experiments
        results = {
            'true_mixtures': [],
            'observed_lambda2': [],
            'predicted_lambda2': [],
            'estimated_mixtures': [],
            'theory_errors': []
        }

        for pi_c in self.config.pi_covariate_range:
            for pi_s in self.config.pi_semantic_range:
                if pi_c + pi_s > 1.0:
                    continue

                trial_results = []

                for trial in range(self.config.n_trials):
                    # Create controlled mixture
                    mixed_features = self.create_controlled_mixture(
                        id_features, cov_features, sem_features, pi_c, pi_s
                    )

                    # Compute spectrum
                    _, lambda2_observed = self.compute_graph_spectrum(mixed_features)

                    # Theoretical prediction
                    lambda2_predicted = self.baseline_lambda2 + pi_c * self.delta_covariate + pi_s * self.delta_semantic

                    # Estimate mixture coefficients
                    pi_c_est, pi_s_est = self.estimate_mixture_coefficients(lambda2_observed, self.baseline_lambda2)

                    trial_results.append({
                        'observed': lambda2_observed,
                        'predicted': lambda2_predicted,
                        'pi_c_est': pi_c_est,
                        'pi_s_est': pi_s_est,
                        'theory_error': abs(lambda2_observed - lambda2_predicted)
                    })

                # Aggregate trial results
                observed_mean = np.mean([r['observed'] for r in trial_results])
                predicted_mean = np.mean([r['predicted'] for r in trial_results])
                pi_c_est_mean = np.mean([r['pi_c_est'] for r in trial_results])
                pi_s_est_mean = np.mean([r['pi_s_est'] for r in trial_results])
                theory_error_mean = np.mean([r['theory_error'] for r in trial_results])

                results['true_mixtures'].append((pi_c, pi_s))
                results['observed_lambda2'].append(observed_mean)
                results['predicted_lambda2'].append(predicted_mean)
                results['estimated_mixtures'].append((pi_c_est_mean, pi_s_est_mean))
                results['theory_errors'].append(theory_error_mean)

                logging.info(f"π_c={pi_c:.1f}, π_s={pi_s:.1f}: "
                           f"Observed λ₂={observed_mean:.4f}, "
                           f"Predicted λ₂={predicted_mean:.4f}, "
                           f"Error={theory_error_mean:.4f}")

        self.results['mixture_experiments'] = results
        return results

    def compute_theoretical_bounds(self) -> Dict:
        """
        Compute theoretical concentration bounds for mixture estimation
        """
        if not self.results['mixture_experiments']:
            return {}

        results = self.results['mixture_experiments']

        # Estimate spectral variance from experiments
        theory_errors = np.array(results['theory_errors'])
        spectral_variance = np.var(theory_errors)

        # Compute theoretical bounds for different sample sizes
        sample_sizes = [100, 500, 1000, 2000, 5000]
        confidence_levels = [0.9, 0.95, 0.99]

        bounds = {}

        if abs(self.delta_covariate - self.delta_semantic) > 1e-6:
            delta = abs(self.delta_covariate - self.delta_semantic)

            for confidence in confidence_levels:
                alpha = 1 - confidence
                bounds[confidence] = {}

                for n in sample_sizes:
                    # From Theorem 2: concentration bound
                    epsilon_bound = np.sqrt(8 * spectral_variance * np.log(2/alpha) / (n * delta**2))
                    bounds[confidence][n] = epsilon_bound

        return {
            'spectral_variance': spectral_variance,
            'separation_gap': abs(self.delta_covariate - self.delta_semantic),
            'concentration_bounds': bounds
        }

    def validate_spectral_theory(self) -> Dict:
        """
        Validate core spectral theory predictions
        """
        if not self.results['mixture_experiments']:
            return {}

        results = self.results['mixture_experiments']

        # Theory-practice correlation
        observed = np.array(results['observed_lambda2'])
        predicted = np.array(results['predicted_lambda2'])

        correlation = np.corrcoef(observed, predicted)[0, 1] if len(observed) > 1 else 0.0
        rmse = np.sqrt(np.mean((observed - predicted)**2))
        relative_error = np.mean(np.abs(observed - predicted) / np.abs(observed + 1e-10))

        # Mixture estimation accuracy
        true_mixtures = np.array(results['true_mixtures'])
        estimated_mixtures = np.array(results['estimated_mixtures'])

        pi_c_error = np.mean(np.abs(true_mixtures[:, 0] - estimated_mixtures[:, 0]))
        pi_s_error = np.mean(np.abs(true_mixtures[:, 1] - estimated_mixtures[:, 1]))

        return {
            'theory_practice_correlation': correlation,
            'rmse': rmse,
            'relative_error': relative_error,
            'mixture_estimation_error': {
                'pi_c_mae': pi_c_error,
                'pi_s_mae': pi_s_error
            },
            'spectral_parameters': {
                'baseline_lambda2': self.baseline_lambda2,
                'delta_covariate': self.delta_covariate,
                'delta_semantic': self.delta_semantic
            }
        }

    def create_validation_plots(self, save_dir: str = './plots'):
        """Create comprehensive validation plots"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        if not self.results['mixture_experiments']:
            logging.warning("No experimental results to plot")
            return

        results = self.results['mixture_experiments']

        # Figure 1: Theory vs Practice
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Scatter plot: Observed vs Predicted λ₂
        ax = axes[0, 0]
        observed = np.array(results['observed_lambda2'])
        predicted = np.array(results['predicted_lambda2'])

        ax.scatter(predicted, observed, alpha=0.7)
        min_val, max_val = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        ax.set_xlabel('Predicted λ₂')
        ax.set_ylabel('Observed λ₂')
        ax.set_title('Theory vs Practice')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mixture estimation accuracy
        ax = axes[0, 1]
        true_mixtures = np.array(results['true_mixtures'])
        estimated_mixtures = np.array(results['estimated_mixtures'])

        ax.scatter(true_mixtures[:, 0], estimated_mixtures[:, 0], alpha=0.7, label='π_c')
        ax.scatter(true_mixtures[:, 1], estimated_mixtures[:, 1], alpha=0.7, label='π_s')
        ax.plot([0, 0.5], [0, 0.5], 'r--', label='Perfect estimation')
        ax.set_xlabel('True mixture coefficient')
        ax.set_ylabel('Estimated mixture coefficient')
        ax.set_title('Mixture Estimation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Theory error distribution
        ax = axes[1, 0]
        theory_errors = np.array(results['theory_errors'])
        ax.hist(theory_errors, bins=20, alpha=0.7, density=True)
        ax.set_xlabel('Theory Error |λ₂_observed - λ₂_predicted|')
        ax.set_ylabel('Density')
        ax.set_title('Theory Error Distribution')
        ax.grid(True, alpha=0.3)

        # Spectral signatures heatmap
        ax = axes[1, 1]
        pi_c_vals = sorted(set([m[0] for m in results['true_mixtures']]))
        pi_s_vals = sorted(set([m[1] for m in results['true_mixtures']]))

        if len(pi_c_vals) > 1 and len(pi_s_vals) > 1:
            heatmap_data = np.zeros((len(pi_s_vals), len(pi_c_vals)))
            for i, (pi_c, pi_s) in enumerate(results['true_mixtures']):
                row = pi_s_vals.index(pi_s)
                col = pi_c_vals.index(pi_c)
                heatmap_data[row, col] = results['observed_lambda2'][i]

            im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(pi_c_vals)))
            ax.set_xticklabels([f'{x:.1f}' for x in pi_c_vals])
            ax.set_yticks(range(len(pi_s_vals)))
            ax.set_yticklabels([f'{x:.1f}' for x in pi_s_vals])
            ax.set_xlabel('π_c (covariate)')
            ax.set_ylabel('π_s (semantic)')
            ax.set_title('Spectral Signatures λ₂(wild)')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/wild_data_validation.png', dpi=150, bbox_inches='tight')
        plt.show()

        logging.info(f"Validation plots saved to {save_dir}/wild_data_validation.png")


def run_complete_wild_data_analysis(model,
                                  id_dataloader,
                                  covariate_dataloader,
                                  semantic_dataloader,
                                  device='cuda'):
    """
    Complete wild data theoretical analysis pipeline

    Usage:
    >>> # Your existing dataloaders
    >>> id_loader = ...  # CIFAR-10 test
    >>> cov_loader = ... # CIFAR-10-C (corruptions)
    >>> sem_loader = ... # SVHN (semantic OOD)
    >>>
    >>> results = run_complete_wild_data_analysis(
    >>>     model, id_loader, cov_loader, sem_loader
    >>> )
    """

    # Initialize analyzer with experimental configuration
    config = WildDataConfig(
        pi_covariate_range=[0.1, 0.2, 0.3, 0.4, 0.5],
        pi_semantic_range=[0.1, 0.2, 0.3],
        k_neighbors=50,
        n_trials=3  # Reduce for faster execution
    )

    analyzer = WildDataTheoreticalAnalyzer(config)

    # Run mixture experiments
    mixture_results = analyzer.run_mixture_experiments(
        model, id_dataloader, covariate_dataloader, semantic_dataloader, device
    )

    # Compute theoretical bounds
    bounds = analyzer.compute_theoretical_bounds()

    # Validate spectral theory
    validation = analyzer.validate_spectral_theory()

    # Create visualization
    analyzer.create_validation_plots()

    # Comprehensive results
    results = {
        'mixture_experiments': mixture_results,
        'theoretical_bounds': bounds,
        'validation_metrics': validation,
        'spectral_parameters': {
            'baseline_lambda2': analyzer.baseline_lambda2,
            'delta_covariate': analyzer.delta_covariate,
            'delta_semantic': analyzer.delta_semantic
        }
    }

    # Print summary
    print("\n" + "="*80)
    print("WILD DATA THEORETICAL ANALYSIS SUMMARY")
    print("="*80)
    print(f"Theory-Practice Correlation: {validation['theory_practice_correlation']:.3f}")
    print(f"RMSE: {validation['rmse']:.6f}")
    print(f"Mixture Estimation Error (π_c): {validation['mixture_estimation_error']['pi_c_mae']:.3f}")
    print(f"Mixture Estimation Error (π_s): {validation['mixture_estimation_error']['pi_s_mae']:.3f}")
    print(f"Baseline λ₂: {analyzer.baseline_lambda2:.6f}")
    print(f"Δ_covariate: {analyzer.delta_covariate:.6f}")
    print(f"Δ_semantic: {analyzer.delta_semantic:.6f}")

    return results


# Example usage for your datasets
if __name__ == "__main__":
    # This is how you'd integrate with your existing code

    # Assuming you have your trained model and dataloaders
    # model = your_trained_model
    # id_loader = your_cifar10_test_loader
    # cov_loader = your_cifar10c_loader  # CIFAR-10-C corruptions
    # sem_loader = your_svhn_loader      # SVHN as semantic OOD

    # results = run_complete_wild_data_analysis(model, id_loader, cov_loader, sem_loader)

    pass