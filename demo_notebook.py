"""
Interactive Demo for Spectral OOD Detection
Run this to see the framework in action with visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import our spectral methods
from spectral_ood_vision import ImageSpectralOODDetector
from advanced_spectral_vision import HybridSpectralOODDetector
from feature_cache import CachedFeatureExtractor

def generate_demo_data():
    """Generate synthetic data to demonstrate spectral OOD detection concepts"""
    
    # ID data: Two well-separated clusters (represents learned manifold)
    np.random.seed(42)
    
    # Create ID data with clear structure
    id_data_1, _ = make_blobs(n_samples=300, centers=[(2, 2), (6, 6)], 
                             cluster_std=0.8, random_state=42)
    id_data_2, _ = make_blobs(n_samples=200, centers=[(4, 1)], 
                             cluster_std=0.6, random_state=43)
    id_data = np.vstack([id_data_1, id_data_2])
    
    # OOD data: Various types of anomalies
    
    # Type 1: Outliers far from ID manifold
    ood_outliers = np.random.uniform(-2, 10, (100, 2))
    mask = ((ood_outliers[:, 0] < 1) | (ood_outliers[:, 0] > 8) | 
            (ood_outliers[:, 1] < 0) | (ood_outliers[:, 1] > 8))
    ood_outliers = ood_outliers[mask][:50]  # Take first 50 that satisfy condition
    
    # Type 2: Bridge points (between ID clusters)
    ood_bridge = np.random.normal([4, 4], [0.3, 0.3], (50, 2))
    
    # Type 3: Different manifold structure
    theta = np.linspace(0, 2*np.pi, 80)
    radius = 1.5
    center = np.array([8, 2])
    ood_circle = np.column_stack([center[0] + radius * np.cos(theta),
                                 center[1] + radius * np.sin(theta)])
    ood_circle += np.random.normal(0, 0.1, ood_circle.shape)  # Add noise
    
    ood_data = np.vstack([ood_outliers, ood_bridge, ood_circle[:30]])  # Limit size
    
    return id_data, ood_data

def demo_basic_spectral_concepts():
    """Demonstrate basic spectral graph theory concepts"""
    
    print("🔬 DEMO 1: Basic Spectral Graph Theory Concepts")
    print("="*60)
    
    # Generate data
    id_data, ood_data = generate_demo_data()
    
    # Visualize data
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Raw data
    axes[0].scatter(id_data[:, 0], id_data[:, 1], c='blue', alpha=0.6, 
                   label='ID Data', s=30)
    axes[0].scatter(ood_data[:, 0], ood_data[:, 1], c='red', alpha=0.6, 
                   label='OOD Data', s=30)
    axes[0].set_title('Raw Data Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Graph construction
    from sklearn.neighbors import kneighbors_graph
    
    # Build graph for ID data
    k_neighbors = 8
    A_id = kneighbors_graph(id_data, n_neighbors=k_neighbors, mode='connectivity')
    A_id = 0.5 * (A_id + A_id.T)
    
    # Visualize graph connections (sample)
    axes[1].scatter(id_data[:, 0], id_data[:, 1], c='blue', alpha=0.6, s=30)
    
    # Show some edges
    edges = np.array(A_id.nonzero()).T
    for i, (u, v) in enumerate(edges[:100]):  # Show first 100 edges
        if u < v:  # Avoid duplicate edges
            axes[1].plot([id_data[u, 0], id_data[v, 0]], 
                        [id_data[u, 1], id_data[v, 1]], 
                        'gray', alpha=0.3, linewidth=0.5)
    
    axes[1].set_title(f'Graph Construction (k={k_neighbors})')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Eigenvalue spectrum
    from scipy.sparse import csr_matrix
    
    # Compute normalized Laplacian
    D = np.array(A_id.sum(axis=1)).flatten()
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
    L_id = np.eye(A_id.shape[0]) - D_sqrt_inv @ A_id.toarray() @ D_sqrt_inv
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(L_id)
    eigenvals = np.sort(eigenvals)[:20]  # First 20 eigenvalues
    
    axes[2].plot(eigenvals, 'bo-', markersize=4)
    axes[2].axvline(x=1, color='red', linestyle='--', alpha=0.7, 
                   label=f'Spectral Gap: {eigenvals[1]-eigenvals[0]:.4f}')
    axes[2].set_xlabel('Eigenvalue Index')
    axes[2].set_ylabel('Eigenvalue')
    axes[2].set_title('Eigenvalue Spectrum')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ ID Data: {len(id_data)} samples, Spectral gap: {eigenvals[1]-eigenvals[0]:.4f}")
    print(f"✅ Algebraic connectivity: {eigenvals[1]:.4f}")
    
    return id_data, ood_data

def demo_spectral_ood_detection():
    """Demonstrate spectral OOD detection methods"""
    
    print("\n🎯 DEMO 2: Spectral OOD Detection Methods")
    print("="*60)
    
    # Generate data
    id_data, ood_data = generate_demo_data()
    
    # Prepare train/test split
    n_train = len(id_data) // 2
    train_data = id_data[:n_train]
    test_id_data = id_data[n_train:]
    
    # Combine test data
    test_data = np.vstack([test_id_data, ood_data])
    test_labels = np.concatenate([np.zeros(len(test_id_data)), 
                                 np.ones(len(ood_data))])
    
    # Test different spectral methods
    methods = {
        'Spectral Gap': ImageSpectralOODDetector(method='spectral_gap', pca_dim=2),
        'Multi-scale': ImageSpectralOODDetector(method='multiscale', pca_dim=2),
        'Unified': ImageSpectralOODDetector(method='unified', pca_dim=2),
        'Hybrid': HybridSpectralOODDetector(embedding_dim=8, pca_components=2, 
                                           ensemble_methods=['spectral_gap', 'heat_kernel'])
    }
    
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    for i, (method_name, detector) in enumerate(methods.items()):
        print(f"Testing {method_name} method...")
        
        try:
            # Fit detector
            detector.fit(train_data)
            
            # Predict scores
            scores = detector.predict_score(test_data)
            
            # Evaluate
            auc = roc_auc_score(test_labels, scores)
            ap = average_precision_score(test_labels, scores)
            
            results[method_name] = {'auc': auc, 'ap': ap, 'scores': scores}
            
            # Plot results
            ax = axes_flat[i]
            
            # Scatter plot with score-based coloring
            id_scores = scores[:len(test_id_data)]
            ood_scores = scores[len(test_id_data):]
            
            scatter1 = ax.scatter(test_id_data[:, 0], test_id_data[:, 1], 
                                c=id_scores, cmap='Blues', alpha=0.7, s=40, 
                                label=f'ID (n={len(test_id_data)})')
            scatter2 = ax.scatter(ood_data[:, 0], ood_data[:, 1], 
                                c=ood_scores, cmap='Reds', alpha=0.7, s=40,
                                label=f'OOD (n={len(ood_data)})')
            
            ax.set_title(f'{method_name}\nAUC: {auc:.3f}, AP: {ap:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter2, ax=ax, label='OOD Score')
            
            print(f"  {method_name}: AUC={auc:.4f}, AP={ap:.4f}")
            
        except Exception as e:
            print(f"  Error with {method_name}: {e}")
            results[method_name] = {'auc': 0, 'ap': 0, 'scores': None}
    
    # ROC curves comparison
    ax_roc = axes_flat[4]
    ax_pr = axes_flat[5]
    
    for method_name, result in results.items():
        if result['scores'] is not None:
            fpr, tpr, _ = roc_curve(test_labels, result['scores'])
            ax_roc.plot(fpr, tpr, label=f"{method_name} (AUC={result['auc']:.3f})")
    
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves Comparison')
    ax_roc.legend()
    ax_roc.grid(True, alpha=0.3)
    
    # Methods comparison bar chart
    method_names = list(results.keys())
    aucs = [results[name]['auc'] for name in method_names]
    aps = [results[name]['ap'] for name in method_names]
    
    x = np.arange(len(method_names))
    width = 0.35
    
    ax_pr.bar(x - width/2, aucs, width, label='AUC', alpha=0.8)
    ax_pr.bar(x + width/2, aps, width, label='Average Precision', alpha=0.8)
    ax_pr.set_xlabel('Method')
    ax_pr.set_ylabel('Score')
    ax_pr.set_title('Method Comparison')
    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels(method_names, rotation=45)
    ax_pr.legend()
    ax_pr.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

def demo_real_vision_example():
    """Demonstrate with real vision data if available"""
    
    print("\n🖼️  DEMO 3: Real Vision Data Example")
    print("="*60)
    
    try:
        from spectral_ood_vision import VisionDatasetLoader, AdvancedFeatureExtractor
        
        # Initialize components
        loader = VisionDatasetLoader('./data', batch_size=32)
        extractor = AdvancedFeatureExtractor(architecture='resnet18')
        
        print("Loading CIFAR-10 and synthetic noise data...")
        
        # Load small subsets
        id_loader = loader.get_cifar10(train=False)
        ood_loader = loader.get_noise_ood(size=200)
        
        # Extract features (small sample)
        id_features, id_labels = extractor.extract_features(id_loader, max_samples=400)
        ood_features, ood_labels = extractor.extract_features(ood_loader, max_samples=200)
        
        print(f"ID features shape: {id_features.shape}")
        print(f"OOD features shape: {ood_features.shape}")
        
        # Train/test split
        n_train = 200
        train_features = id_features[:n_train]
        test_id_features = id_features[n_train:300]
        
        test_features = np.vstack([test_id_features, ood_features])
        test_labels = np.concatenate([np.zeros(len(test_id_features)), 
                                     np.ones(len(ood_features))])
        
        # Test hybrid detector
        detector = HybridSpectralOODDetector(
            embedding_dim=32,
            pca_components=128,
            adaptive_k=True,
            ensemble_methods=['spectral_gap', 'heat_kernel']
        )
        
        print("Training hybrid spectral detector...")
        detector.fit(train_features)
        
        print("Computing OOD scores...")
        scores = detector.predict_score(test_features)
        
        # Evaluate
        auc = roc_auc_score(test_labels, scores)
        ap = average_precision_score(test_labels, scores)
        
        print(f"✅ Results on real vision data:")
        print(f"   AUC: {auc:.4f}")
        print(f"   Average Precision: {ap:.4f}")
        
        # Visualize with PCA
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PCA visualization
        pca = PCA(n_components=2)
        test_features_2d = pca.fit_transform(test_features)
        
        id_2d = test_features_2d[:len(test_id_features)]
        ood_2d = test_features_2d[len(test_id_features):]
        
        axes[0].scatter(id_2d[:, 0], id_2d[:, 1], c='blue', alpha=0.6, 
                       label='CIFAR-10 (ID)', s=20)
        axes[0].scatter(ood_2d[:, 0], ood_2d[:, 1], c='red', alpha=0.6, 
                       label='Noise (OOD)', s=20)
        axes[0].set_title('PCA Visualization of Feature Space')
        axes[0].set_xlabel('First Principal Component')
        axes[0].set_ylabel('Second Principal Component')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Score distribution
        id_scores = scores[:len(test_id_features)]
        ood_scores = scores[len(test_id_features):]
        
        axes[1].hist(id_scores, bins=20, alpha=0.7, label='ID Scores', 
                    color='blue', density=True)
        axes[1].hist(ood_scores, bins=20, alpha=0.7, label='OOD Scores', 
                    color='red', density=True)
        axes[1].set_xlabel('OOD Score')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Score Distributions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Real vision demo failed (likely missing data): {e}")
        print("Skipping real vision demonstration...")

def main_demo():
    """Run the complete interactive demo with argparse support"""
    parser = argparse.ArgumentParser(description='Interactive Spectral OOD Detection Demo')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing datasets (default: ./data)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Directory for caching features (default: ./cache)')
    parser.add_argument('--skip_real_data', action='store_true',
                       help='Skip real vision data demo')
    parser.add_argument('--cache_info', action='store_true',
                       help='Show cache information and exit')
    
    args = parser.parse_args()
    
    print("🚀 SPECTRAL OOD DETECTION - INTERACTIVE DEMO")
    print("="*80)
    print("This demo showcases spectral graph theory for OOD detection")
    print("across synthetic and real computer vision data.")
    print(f"Data Directory: {args.data_dir}")
    print(f"Cache Directory: {args.cache_dir}")
    print("\n")
    
    # Initialize cache system
    cache_system = CachedFeatureExtractor(cache_dir=args.cache_dir)
    
    # Handle cache info
    if args.cache_info:
        cache_system.print_cache_info()
        return
    
    # Demo 1: Basic concepts
    id_data, ood_data = demo_basic_spectral_concepts()
    
    # Demo 2: OOD detection methods
    results = demo_spectral_ood_detection()
    
    # Demo 3: Real vision data (optional)
    if not args.skip_real_data:
        demo_real_vision_example()
    else:
        print("\n🖼️  Real vision demo skipped (use --skip_real_data to enable)")
    
    # Summary
    print("\n🎉 DEMO SUMMARY")
    print("="*60)
    print("You've seen how spectral graph theory can be used for OOD detection:")
    print("1. 📊 Graph construction from data points")
    print("2. 🔍 Eigenvalue analysis reveals structural properties")  
    print("3. 🎯 Multiple spectral methods for robust detection")
    print("4. 🖼️  Application to high-dimensional vision features")
    print("\nKey advantages of spectral methods:")
    print("• Theoretical guarantees from graph theory")
    print("• Captures global and local structural properties")
    print("• Robust across different data distributions")
    print("• Scales well to high-dimensional feature spaces")
    
    if results:
        best_method = max(results.keys(), key=lambda k: results[k]['auc'])
        best_auc = results[best_method]['auc']
        print(f"\n🏆 Best performing method: {best_method} (AUC: {best_auc:.4f})")
    
    print(f"\n📚 For more details, see the full implementation in:")
    print(f"   - spectral_ood_vision.py (core methods)")
    print(f"   - advanced_spectral_vision.py (hybrid approaches)")
    print(f"   - run_comprehensive_experiment.py (full evaluation)")

if __name__ == "__main__":
    main_demo()