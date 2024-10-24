import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
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

import wandb

class SimpleGNNforOOD(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(SimpleGNNforOOD, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        logits = self.classifier(x)
        return logits, x  # Return both logits and embeddings

def contrastive_loss(embeddings, labels, temperature=0.5):
    batch_size = embeddings.size(0)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(embeddings.device)

    contrast_feature = embeddings.contiguous().view(batch_size, -1)
    anchor_dot_contrast = torch.div(
        torch.matmul(contrast_feature, contrast_feature.T),
        temperature)

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mask = mask.repeat(2, 2)  # Repeat for both ID and OOD samples
    mask.fill_diagonal_(0)

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    loss = - mean_log_prob_pos
    loss = loss.view(2, batch_size).mean()

    return loss

def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def train_epoch(model, train_loader, ood_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for (id_data, ood_data) in zip(train_loader, ood_loader):
        optimizer.zero_grad()

        # Process ID data
        id_data = id_data.to(device)
        id_logits, id_embeddings = model(id_data)
        id_loss = F.cross_entropy(id_logits, id_data.y)

        # Process OOD data
        ood_data = ood_data.to(device)
        ood_logits, ood_embeddings = model(ood_data)

        # Combine embeddings and create labels for contrastive loss
        combined_embeddings = torch.cat([id_embeddings, ood_embeddings], dim=0)
        labels = torch.cat([torch.zeros(id_embeddings.size(0)), torch.ones(ood_embeddings.size(0))], dim=0).to(device)

        # Compute contrastive loss
        cont_loss = contrastive_loss(combined_embeddings, labels)

        # Total loss
        loss = id_loss + cont_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = id_logits.argmax(dim=1)
        correct += (pred == id_data.y).sum().item()
        total += id_data.y.size(0)

    return total_loss / len(train_loader), correct / total

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            logits, _ = model(data)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            all_probs.append(probs)
            all_labels.append(data.y)

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    entropies = entropy(all_probs)

    return correct / total, entropies, all_probs, all_labels

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
    wandb.init(project="simple-gcn-ood", name="experiment-1")
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.config.update({
        "num_features": 512,
        "hidden_dim": 128,
        "num_classes": 10,
        "num_prototypes": 100,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "num_epochs": 50
    })
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

    model = SimpleGNNforOOD(num_features=512, hidden_dim=128, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 50
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, cifar_train_loader, cifar_test_loader, optimizer, device)
        val_acc, val_entropies, val_probs, val_labels = evaluate(model, cifar_test_loader, device)
        ood_acc, ood_entropies, ood_probs, ood_labels = evaluate(model, svhn_test_loader, device)

        val_loss = F.cross_entropy(val_probs, val_labels)
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # OOD Detection metrics
        val_entropy_mean = val_entropies.mean().item()
        ood_entropy_mean = ood_entropies.mean().item()
        print(f'Val Entropy: {val_entropy_mean:.4f}, OOD Entropy: {ood_entropy_mean:.4f}')

        # Combine ID and OOD data for AUROC calculation
        combined_entropies = torch.cat([val_entropies, ood_entropies]).cpu().numpy()
        combined_labels = torch.cat([torch.zeros_like(val_labels), torch.ones_like(ood_labels)]).cpu().numpy()
        auroc = roc_auc_score(combined_labels, combined_entropies)
        print(f'AUROC: {auroc:.4f}')

        # Calculate FPR at 95% TPR
        fpr, tpr, thresholds = roc_curve(combined_labels, combined_entropies)
        idx = np.argmin(np.abs(tpr - 0.95))
        fpr_at_95_tpr = fpr[idx]

        # Calculate Expected Calibration Error
        ece = expected_calibration_error(combined_labels, combined_entropies)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss.item(),
            "val_acc": val_acc,
            "val_entropy": val_entropy_mean,
            "ood_entropy": ood_entropy_mean,
            "auroc": auroc,
            "fpr_at_95_tpr": fpr_at_95_tpr,
            "ece": ece,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Val Entropy: {val_entropy_mean:.4f}, OOD Entropy: {ood_entropy_mean:.4f}, AUROC: {auroc:.4f}')

        # Generate plots every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Reliability Diagram
            reliability_diagram(combined_labels, combined_entropies)
            wandb.log({"reliability_diagram": wandb.Image("reliability_diagram.png")})

            # Precision-Recall Curve
            plot_precision_recall_curve(combined_labels, combined_entropies)
            wandb.log({"precision_recall_curve": wandb.Image("precision_recall_curve.png")})

            # Confidence Distributions
            plot_confidence_distributions(val_entropies.cpu().numpy(), ood_entropies.cpu().numpy())
            wandb.log({"confidence_distributions": wandb.Image("confidence_distributions.png")})

            # ROC Curve
            plot_roc_curve(combined_labels, combined_entropies, "ROC Curve for OOD Detection")
            wandb.log({"roc_curve": wandb.Image("ROC Curve for OOD Detection.png")})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.save('best_model.pth')

    wandb.finish()

if __name__ == "__main__":
    main()

