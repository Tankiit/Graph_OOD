import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return trainset

def image_to_graph(image, k=8):
    """Convert an image to a graph based on pixel similarity."""
    image_flat = image.reshape(-1, 3)
    adj_matrix = kneighbors_graph(image_flat, k, mode='distance', include_self=True)
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    return G

def compute_negative_laplacian(G):
    """Compute the negative Laplacian of a graph."""
    L = nx.laplacian_matrix(G).todense()
    return -L

def extract_spectral_features(NL, n_features=10):
    """Extract spectral features from the negative Laplacian."""
    eigenvalues, eigenvectors = eigh(NL)
    return eigenvalues[-n_features:], eigenvectors[:, -n_features:]

def create_synthetic_ood(NL, perturbation_strength=0.2):
    """Create a synthetic OOD sample by perturbing the negative Laplacian."""
    noise = np.random.randn(*NL.shape) * perturbation_strength
    return NL + noise

def analyze_cifar10():
    trainset = load_cifar10()
    spectral_features = []
    labels = []

    for i in range(1000):  # Analyze a subset for computational efficiency
        image, label = trainset[i]
        G = image_to_graph(image.numpy())
        NL = compute_negative_laplacian(G)
        eigenvalues, _ = extract_spectral_features(NL)
        spectral_features.append(eigenvalues)
        labels.append(label)

    # Create synthetic OOD samples
    num_ood_samples = 100
    ood_features = []
    for _ in range(num_ood_samples):
        NL = compute_negative_laplacian(image_to_graph(np.random.rand(32, 32, 3)))
        NL_ood = create_synthetic_ood(NL)
        eigenvalues, _ = extract_spectral_features(NL_ood)
        ood_features.append(eigenvalues)
    
    spectral_features.extend(ood_features)
    labels.extend([10] * num_ood_samples)  # 10 is the label for OOD class

    return np.array(spectral_features), np.array(labels)

def visualize_spectral_features(features, labels):
    tsne = TSNE(n_components=2, random_state=42)
    embedded_features = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Spectral Features (including OOD)')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig('cifar10_spectral_features_with_ood.png')
    plt.close()

def plot_eigenvalue_distributions(features, labels):
    plt.figure(figsize=(12, 8))
    for i in range(11):  # 10 CIFAR-10 classes + 1 OOD class
        class_features = features[labels == i]
        plt.plot(np.mean(class_features, axis=0), label=f'Class {i}')
    plt.title('Average Eigenvalue Distribution by Class')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.savefig('cifar10_eigenvalue_distributions.png')
    plt.close()

# Main execution
spectral_features, labels = analyze_cifar10()
visualize_spectral_features(spectral_features, labels)
plot_eigenvalue_distributions(spectral_features, labels)

# Analyze separability
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(spectral_features, labels, test_size=0.3, random_state=42)

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(classification_report(y_test, y_pred))