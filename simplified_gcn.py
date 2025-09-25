import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, models
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
    
    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)

class SimplifiedGCNforOOD(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(SimplifiedGCNforOOD, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=3, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        logits = self.classifier(x)
        return x, logits
    
def create_outlier_samples(images, labels, outlier_ratio=0.2):
    num_samples = len(images)
    num_outliers = int(num_samples * outlier_ratio)
    
    # Randomly select indices for outliers
    outlier_indices = np.random.choice(num_samples, num_outliers, replace=False)
    
    # Create new images and labels tensors
    new_images = images.clone()
    new_labels = labels.clone()
    
    # Replace selected samples with random noise and assign outlier label (10)
    new_images[outlier_indices] = torch.rand_like(new_images[outlier_indices])
    new_labels[outlier_indices] = 10  # 10 is the outlier class
    
    return new_images, new_labels

def create_frustrated_graph(features, labels, feature_extractor, device, frustration_threshold=0.5):
    feature_extractor.eval()
    with torch.no_grad():
        extracted_features = feature_extractor(features.to(device)).cpu()
    
    graphs = []
    for i in range(extracted_features.size(0)):
        x = extracted_features[i].unsqueeze(0)
        y = labels[i].unsqueeze(0)
        
        # Create fully connected graph
        num_nodes = x.size(0)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        
        # Assign edge signs based on feature similarity
        edge_attr = []
        for j in range(edge_index.size(1)):
            u, v = edge_index[:, j]
            similarity = F.cosine_similarity(x[u], x[v], dim=0)
            sign = 1.0 if similarity > frustration_threshold else -1.0
            edge_attr.append(sign)
        
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    
    return Batch.from_data_list(graphs)

def train_epoch(model, train_loader, feature_extractor, optimizer, device, outlier_ratio=0.2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        # Create outlier samples
        images, labels = create_outlier_samples(images, labels, outlier_ratio)

        # Create frustrated graph datasets
        graph_data = create_frustrated_graph(images, labels, feature_extractor, device)
        graph_data = graph_data.to(device)

        # Forward pass
        _, logits = model(graph_data)

        # Classification loss
        loss = F.cross_entropy(logits, graph_data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == graph_data.y).sum().item()
        total += graph_data.y.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, data_loader, feature_extractor, device, is_ood=False):
    model.eval()
    all_features = []
    all_scores = []
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            if is_ood:
                labels = torch.full_like(labels, 10)  # Assign OOD label
            graph_data = create_frustrated_graph(images, labels, feature_extractor, device)
            graph_data = graph_data.to(device)
            features, logits = model(graph_data)
            loss = F.cross_entropy(logits, graph_data.y)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            all_features.extend(features.cpu().numpy())
            all_scores.extend(F.softmax(logits, dim=1)[:, :10].max(dim=1)[0].cpu().numpy())  # Max prob of ID classes
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(graph_data.y.cpu().numpy())

    return np.array(all_features), np.array(all_scores), np.array(all_preds), np.array(all_labels), total_loss / len(data_loader)

def reliability_diagram(true_labels, pred_probs, num_bins=10):
    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(pred_probs, bins, right=True)
    
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    for b in range(num_bins):
        mask = indices == b + 1
        bin_accuracies[b] = np.mean(true_labels[mask] == 1)
        bin_confidences[b] = np.mean(pred_probs[mask])
        bin_counts[b] = np.sum(mask)
    
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.bar(bins[:-1], bin_accuracies, width=bin_size, alpha=0.3, align='edge')
    plt.plot(bin_confidences, bin_accuracies, '-o')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.savefig('reliability_diagram.png')
    plt.close()

def plot_precision_recall_curve(true_labels, pred_probs):
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    avg_precision = average_precision_score(true_labels, pred_probs)
    
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision, label=f'AP={avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('precision_recall_curve.png')
    plt.close()

def plot_confidence_distributions(in_dist_probs, ood_probs):
    plt.figure(figsize=(10, 6))
    plt.hist(in_dist_probs, bins=50, alpha=0.5, label='In-Distribution')
    plt.hist(ood_probs, bins=50, alpha=0.5, label='OOD')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.savefig('confidence_distributions.png')
    plt.close()

def calculate_ood_metrics(in_scores, out_scores):
    # Combine in-distribution and out-of-distribution scores
    scores = np.concatenate([in_scores, out_scores])
    labels = np.concatenate([np.ones_like(in_scores), np.zeros_like(out_scores)])
    
    # Calculate AU-ROC
    auroc = roc_auc_score(labels, scores)
    
    # Calculate FPR at 95% TPR
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr95_idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_tpr95 = fpr[tpr95_idx]
    
    return auroc, fpr_at_tpr95

def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{title}.png')

def plot_roc_curve(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f'{title}.png')
    plt.close()

def expected_calibration_error(true_labels, pred_probs, num_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
    true_labels: true labels (binary: 1 for in-distribution, 0 for OOD)
    pred_probs: predicted probabilities
    num_bins: number of bins for the calculation
    
    Returns:
    ece: Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = true_labels
    confidences = pred_probs
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find the confidences in this bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CIFAR-10 (in-distribution) and SVHN (OOD) datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    # Data Loaders
    cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=64, shuffle=True)
    cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=64, shuffle=False)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size=64, shuffle=False)

    feature_extractor = FeatureExtractor().to(device)
    model = SimplifiedGCNforOOD(num_features=512, hidden_dim=128, num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 50
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, cifar_train_loader, feature_extractor, optimizer, device)
        _,val_scores, val_preds, val_labels, val_loss, = evaluate(model, cifar_test_loader, feature_extractor, device)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Load best model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluation
    cifar_features,cifar_scores, cifar_preds, cifar_labels, _ = evaluate(model, cifar_test_loader, feature_extractor, device)
    svhn_features,svhn_scores, svhn_preds, svhn_labels, _ = evaluate(model, svhn_test_loader, feature_extractor, device, is_ood=True)

    # OOD Detection
    labels = np.concatenate([np.ones_like(cifar_scores), np.zeros_like(svhn_scores)])
    scores = np.concatenate([cifar_scores, svhn_scores])

    auroc, fpr_at_tpr95 = calculate_ood_metrics(cifar_scores, svhn_scores)
    print(f"AUROC score for OOD detection: {auroc:.4f}")
    print(f"FPR at 95% TPR for OOD detection: {fpr_at_tpr95:.4f}")

    plot_roc_curve(labels, scores, "OOD Detection ROC Curve")
    plot_precision_recall_curve(labels, scores)
    plot_confidence_distributions(cifar_scores, svhn_scores)
    reliability_diagram(labels, scores)
    ece = expected_calibration_error(labels, scores)
    print(f"Expected Calibration Error: {ece:.4f}")

    # Plot confusion matrix
    all_preds = np.concatenate([cifar_preds, svhn_preds])
    all_labels = np.concatenate([cifar_labels, svhn_labels])
    class_names = [f'Class {i}' for i in range(10)] + ['OOD']
    plot_confusion_matrix(all_labels, all_preds, class_names, "Confusion Matrix")

if __name__ == "__main__":
    main()