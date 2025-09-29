"""
Visual intuition for why spectral gaps work for OOD detection
Demonstrates the connection between graph structure and eigenvalues
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh
import networkx as nx

def create_toy_example():
    """Create toy datasets to demonstrate spectral gap intuition"""
    
    # ID data: Well-separated clusters (like CIFAR classes)
    np.random.seed(42)
    X_id, y_id = make_blobs(n_samples=200, centers=4, n_features=2, 
                           center_box=(-5, 5), cluster_std=1.2)
    
    # OOD data: Different distribution (like SVHN)
    X_ood = np.random.uniform(-8, 8, (50, 2))  # Uniform random points
    
    # Combined data
    X_combined = np.vstack([X_id, X_ood])
    
    return X_id, X_ood, X_combined, y_id

def build_adjacency_and_analyze(X, k=5):
    """Build k-NN adjacency matrix and compute spectral properties"""
    
    # Build k-NN graph
    adj_sparse = kneighbors_graph(X, n_neighbors=k, mode='connectivity')
    adj_matrix = 0.5 * (adj_sparse + adj_sparse.T).toarray()
    
    # Compute Laplacian
    degrees = np.sum(adj_matrix, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))
    laplacian = np.eye(len(X)) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    
    # Compute eigenvalues
    try:
        eigenvals = eigsh(laplacian, k=min(10, laplacian.shape[0]-1), 
                         which='SM', return_eigenvectors=False)
    except:
        eigenvals = np.linalg.eigvals(laplacian)
        eigenvals = np.sort(eigenvals)[:10]
    
    eigenvals = np.sort(eigenvals)
    
    return adj_matrix, eigenvals

def visualize_spectral_intuition():
    """Visualize why spectral gaps change between ID and OOD"""
    
    X_id, X_ood, X_combined, y_id = create_toy_example()
    
    # Analyze ID data
    adj_id, eigs_id = build_adjacency_and_analyze(X_id)
    
    # Analyze combined data  
    adj_combined, eigs_combined = build_adjacency_and_analyze(X_combined)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Data distributions
    axes[0, 0].scatter(X_id[:, 0], X_id[:, 1], c=y_id, cmap='viridis', 
                      alpha=0.7, s=50, label='ID Data (CIFAR-like)')
    axes[0, 0].scatter(X_ood[:, 0], X_ood[:, 1], c='red', marker='x', 
                      s=80, alpha=0.8, label='OOD Data (SVHN-like)')
    axes[0, 0].set_title('Data Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Adjacency matrices
    im1 = axes[0, 1].imshow(adj_id, cmap='Blues', interpolation='nearest')
    axes[0, 1].set_title('ID Data Adjacency Matrix')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Sample Index')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(adj_combined, cmap='Blues', interpolation='nearest')
    axes[0, 2].set_title('Combined (ID + OOD) Adjacency Matrix')
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('Sample Index')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Plot 3: Eigenvalue spectra
    axes[1, 0].plot(eigs_id, 'bo-', label='ID Data', markersize=8, linewidth=2)
    axes[1, 0].plot(eigs_combined, 'rs-', label='Combined Data', markersize=8, linewidth=2)
    axes[1, 0].set_xlabel('Eigenvalue Index')
    axes[1, 0].set_ylabel('Eigenvalue')
    axes[1, 0].set_title('Eigenvalue Spectrum Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Highlight spectral gap
    if len(eigs_id) > 1:
        axes[1, 0].axhspan(eigs_id[0], eigs_id[1], alpha=0.2, color='blue', 
                          label=f'ID Spectral Gap: {eigs_id[1] - eigs_id[0]:.3f}')
    if len(eigs_combined) > 1:
        axes[1, 0].axhspan(eigs_combined[0], eigs_combined[1], alpha=0.2, color='red',
                          label=f'Combined Spectral Gap: {eigs_combined[1] - eigs_combined[0]:.3f}')
    
    # Plot 4: Spectral gap comparison
    datasets = ['ID Only', 'ID + OOD']
    spectral_gaps = [eigs_id[1] - eigs_id[0] if len(eigs_id) > 1 else 0,
                    eigs_combined[1] - eigs_combined[0] if len(eigs_combined) > 1 else 0]
    
    bars = axes[1, 1].bar(datasets, spectral_gaps, color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('Spectral Gap (λ₁ - λ₀)')
    axes[1, 1].set_title('Spectral Gap Changes with OOD')
    
    # Add value labels
    for bar, gap in zip(bars, spectral_gaps):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(spectral_gaps)*0.02,
                       f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Connectivity analysis
    lambda1_values = [eigs_id[1] if len(eigs_id) > 1 else 0,
                     eigs_combined[1] if len(eigs_combined) > 1 else 0]
    
    axes[1, 2].bar(datasets, lambda1_values, color=['cyan', 'orange'], alpha=0.7)
    axes[1, 2].set_ylabel('λ₁ (Algebraic Connectivity)')
    axes[1, 2].set_title('How OOD Affects Graph Connectivity')
    
    # Add annotations
    connectivity_change = lambda1_values[1] - lambda1_values[0]
    change_text = f'Change: {connectivity_change:+.3f}'
    axes[1, 2].text(0.5, max(lambda1_values) * 0.8, change_text, 
                   ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return eigs_id, eigs_combined

def explain_spectral_theory():
    """Explain the theoretical connection"""
    
    print("Why Spectral Gaps Work for OOD Detection:")
    print("=" * 50)
    print()
    
    print("1. GRAPH STRUCTURE REFLECTS DATA DISTRIBUTION")
    print("   • ID data: Similar samples connect → coherent clusters")
    print("   • OOD data: Different patterns → disrupted connectivity")
    print()
    
    print("2. SPECTRAL GAP = CONNECTIVITY SIGNATURE")
    print("   • λ₀ = 0 always (connected components)")
    print("   • λ₁ = algebraic connectivity (how well-connected)")
    print("   • Gap λ₁ - λ₀ = structural coherence measure")
    print()
    
    print("3. CHEEGER'S INEQUALITY PROVIDES BOUNDS")
    print("   • λ₁/2 ≤ h(G) ≤ √(2λ₁)")
    print("   • h(G) = min cut ratio (bottleneckedness)")
    print("   • Larger λ₁ → better connectivity → smaller h(G)")
    print()
    
    print("4. OOD DETECTION MECHANISM")
    print("   • Train on ID data → learn reference spectral signature")
    print("   • Test data changes graph structure → changes eigenvalues")
    print("   • Spectral gap deviation indicates distribution shift")
    print()
    
    print("5. MATHEMATICAL GUARANTEES")
    print("   • Matrix perturbation theory bounds eigenvalue changes")
    print("   • Davis-Kahan theorem: ||sin θ|| ≤ 2||Δ||/gap")
    print("   • Concentration inequalities provide statistical guarantees")

def demonstrate_real_vs_synthetic():
    """Show why this works better than random features"""
    
    print("\nWhy Spectral Analysis > Random Features:")
    print("=" * 45)
    
    # Create structured vs random data
    X_structured, _ = make_blobs(n_samples=100, centers=3, n_features=50, cluster_std=1.0)
    X_random = np.random.randn(100, 50)
    
    # Analyze both
    _, eigs_structured = build_adjacency_and_analyze(X_structured)
    _, eigs_random = build_adjacency_and_analyze(X_random)
    
    structured_gap = eigs_structured[1] - eigs_structured[0] if len(eigs_structured) > 1 else 0
    random_gap = eigs_random[1] - eigs_random[0] if len(eigs_random) > 1 else 0
    
    print(f"Structured data spectral gap: {structured_gap:.4f}")
    print(f"Random data spectral gap:     {random_gap:.4f}")
    print(f"Ratio (structured/random):    {structured_gap/max(random_gap, 1e-8):.2f}")
    print()
    print("→ Structured data has more distinctive spectral signatures!")
    print("→ This is why CIFAR vs SVHN should show clear differences")

if __name__ == "__main__":
    print("Spectral Gap Intuition for OOD Detection")
    print("========================================")
    
    # Run theoretical explanation
    explain_spectral_theory()
    
    # Show visual intuition
    print("\nGenerating visual demonstration...")
    eigs_id, eigs_combined = visualize_spectral_intuition()
    
    # Print results
    print(f"\nResults from toy example:")
    print(f"ID data λ₁: {eigs_id[1]:.4f}")
    print(f"Combined λ₁: {eigs_combined[1]:.4f}")
    print(f"Change: {eigs_combined[1] - eigs_id[1]:+.4f}")
    
    if eigs_combined[1] > eigs_id[1]:
        print("OOD data increases λ₁ (changes connectivity)")
    else:
        print("OOD data decreases λ₁ (fragments structure)")
    
    
    # Show structured vs random comparison
    demonstrate_real_vs_synthetic()