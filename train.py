"""
Main Training Pipeline for Graph-based Spectral OOD Detection

This is the main entry point for training models and evaluating OOD detection methods.
It integrates:
1. Model training with CIFAR datasets
2. Energy-based OOD detection (Wang & Li)
3. Spectral analysis with k-NN adjacency matrices
4. Comprehensive evaluation across multiple OOD datasets

Usage:
python train.py --model wrn --dataset cifar10 --epochs 200 --ood_eval
"""

import argparse
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

# Import our custom modules
from models.wrn import WideResNet
from models.resnet import ResNet
from dataloader.cifar10 import CIFAR10DataLoader
from dataloader.cifar100 import CIFAR100DataLoader
from utils.spectral_monitor import SpectralMonitor
from utils.wang_li_energy import WangLiEnergyDetector
from ood_postprocessors.spectral_schemanet import SpectralSchemaNetDetector

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# ============================================================================
# MODEL FACTORY
# ============================================================================

def get_model(model_name, num_classes, device='cuda'):
    """Factory function to create models"""
    if model_name.lower() == 'wrn':
        model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes)
    elif model_name.lower() == 'resnet':
        model = ResNet(depth=50, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    return model

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, logger):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            logger.info(f'Batch {batch_idx}/{len(train_loader)}, '
                       f'Loss: {loss.item():.4f}, '
                       f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# ============================================================================
# OOD EVALUATION
# ============================================================================

def evaluate_ood_detection(model, train_loader, id_test_loader, ood_loaders, device, logger):
    """Comprehensive OOD evaluation"""
    logger.info("Starting OOD evaluation...")
    
    results = {}
    
    # Initialize OOD detectors
    energy_detector = WangLiEnergyDetector(model)
    spectral_detector = SpectralSchemaNetDetector(model)
    spectral_monitor = SpectralMonitor()
    
    # Extract training features for spectral analysis
    logger.info("Extracting training features...")
    train_features, train_labels = energy_detector.extract_features(train_loader, device)
    
    # Build spectral graph
    logger.info("Building spectral graph...")
    spectral_monitor.build_graph(train_features, train_labels, k=10)
    
    # Evaluate on each OOD dataset
    for ood_name, ood_loader in ood_loaders.items():
        logger.info(f"Evaluating on {ood_name}...")
        
        # Energy-based detection
        id_energy_scores = energy_detector.compute_energy_scores(id_test_loader, device)
        ood_energy_scores = energy_detector.compute_energy_scores(ood_loader, device)
        
        # Spectral-based detection
        id_spectral_scores = spectral_detector.compute_spectral_scores(
            id_test_loader, spectral_monitor, device
        )
        ood_spectral_scores = spectral_detector.compute_spectral_scores(
            ood_loader, spectral_monitor, device
        )
        
        # Compute metrics
        energy_auc = compute_auc(id_energy_scores, ood_energy_scores, reverse=True)
        spectral_auc = compute_auc(id_spectral_scores, ood_spectral_scores)
        
        results[ood_name] = {
            'energy_auc': energy_auc,
            'spectral_auc': spectral_auc,
            'id_energy_mean': np.mean(id_energy_scores),
            'ood_energy_mean': np.mean(ood_energy_scores),
            'id_spectral_mean': np.mean(id_spectral_scores),
            'ood_spectral_mean': np.mean(ood_spectral_scores)
        }
        
        logger.info(f"{ood_name} - Energy AUC: {energy_auc:.4f}, Spectral AUC: {spectral_auc:.4f}")
    
    return results

def compute_auc(id_scores, ood_scores, reverse=False):
    """Compute AUC for OOD detection"""
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    if reverse:
        y_scores = -y_scores  # Reverse for energy scores (lower = more OOD)
    
    return roc_auc_score(y_true, y_scores)

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Graph-based Spectral OOD Detection Training')
    parser.add_argument('--model', type=str, default='wrn', choices=['wrn', 'resnet'],
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--save_every', type=int, default=25, help='Save checkpoint every N epochs')
    parser.add_argument('--ood_eval', action='store_true', help='Run OOD evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.output_dir)
    
    logger.info(f"Training {args.model} on {args.dataset} for {args.epochs} epochs")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.output_dir}/plots", exist_ok=True)
    
    # Load data
    if args.dataset == 'cifar10':
        num_classes = 10
        data_loader = CIFAR10DataLoader(args.data_dir, batch_size=args.batch_size)
    else:
        num_classes = 100
        data_loader = CIFAR100DataLoader(args.data_dir, batch_size=args.batch_size)
    
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    
    # Create model
    model = get_model(args.model, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, logger)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        test_accs.append(test_acc)
        
        # Update scheduler
        scheduler.step()
        
        logger.info(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, '
                   f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or test_acc > best_acc:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_acc': test_acc,
                'train_loss': train_loss,
                'train_acc': train_acc
            }
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(checkpoint, f"{args.output_dir}/checkpoints/best_model.pt")
                logger.info(f'New best model saved with accuracy: {test_acc:.2f}%')
            
            torch.save(checkpoint, f"{args.output_dir}/checkpoints/checkpoint_epoch_{epoch+1}.pt")
    
    # Save training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/plots/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f'Training completed. Best test accuracy: {best_acc:.2f}%')
    
    # OOD Evaluation
    if args.ood_eval:
        logger.info("Starting OOD evaluation...")
        
        # Get OOD datasets
        ood_loaders = {}
        if args.dataset == 'cifar10':
            # CIFAR-10 as ID, CIFAR-100 as OOD
            cifar100_loader = CIFAR100DataLoader(args.data_dir, batch_size=args.batch_size)
            ood_loaders['cifar100'] = cifar100_loader.get_test_loader()
            
            # Add more OOD datasets as needed
            # ood_loaders['svhn'] = get_svhn_loader(args.data_dir, args.batch_size)
            # ood_loaders['textures'] = get_textures_loader(args.data_dir, args.batch_size)
        
        # Run OOD evaluation
        ood_results = evaluate_ood_detection(model, train_loader, test_loader, ood_loaders, device, logger)
        
        # Save results
        with open(f"{args.output_dir}/ood_results.json", 'w') as f:
            json.dump(ood_results, f, indent=2)
        
        # Create results summary
        logger.info("OOD Detection Results:")
        for dataset, metrics in ood_results.items():
            logger.info(f"{dataset}:")
            logger.info(f"  Energy AUC: {metrics['energy_auc']:.4f}")
            logger.info(f"  Spectral AUC: {metrics['spectral_auc']:.4f}")
    
    logger.info("Training pipeline completed successfully!")

if __name__ == '__main__':
    main()
