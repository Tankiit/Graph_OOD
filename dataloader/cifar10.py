"""
CIFAR-10 DataLoader for Graph-based Spectral OOD Detection

This module provides comprehensive data loading capabilities for CIFAR-10 dataset
with support for various augmentations and preprocessing techniques.

Key features:
1. Standard and advanced data augmentations
2. Support for different normalization schemes
3. Custom transforms for OOD detection
4. Efficient data loading with caching
5. Support for subset creation and sampling
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import logging


class CIFAR10Dataset(Dataset):
    """
    Custom CIFAR-10 dataset with enhanced functionality
    """
    
    def __init__(self, root, train=True, transform=None, download=True):
        """
        Args:
            root (str): Root directory of dataset
            train (bool): If True, creates dataset from training set, otherwise test set
            transform (callable, optional): A function/transform that takes in PIL Image
                and returns a transformed version
            download (bool): If True, downloads the dataset from the internet
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        
        # Load CIFAR-10 dataset
        self.dataset = CIFAR10(root=root, train=train, download=download)
        self.data = self.dataset.data
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[idx], self.targets[idx]
        
        # Convert to PIL Image
        img = torchvision.transforms.ToPILImage()(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def get_class_samples(self, class_idx, num_samples=None):
        """
        Get samples from a specific class
        
        Args:
            class_idx (int): Class index
            num_samples (int, optional): Number of samples to return
            
        Returns:
            list: List of indices for the specified class
        """
        class_indices = [i for i, target in enumerate(self.targets) if target == class_idx]
        
        if num_samples is not None:
            np.random.seed(42)  # For reproducibility
            class_indices = np.random.choice(class_indices, 
                                           min(num_samples, len(class_indices)), 
                                           replace=False).tolist()
        
        return class_indices


class CIFAR10DataLoader:
    """
    Comprehensive CIFAR-10 data loader with multiple augmentation strategies
    """
    
    def __init__(self, root_dir, batch_size=128, num_workers=4, download=True):
        """
        Args:
            root_dir (str): Root directory for dataset
            batch_size (int): Batch size for data loaders
            num_workers (int): Number of worker processes for data loading
            download (bool): Whether to download dataset if not present
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        
        # Create transforms
        self._create_transforms()
        
        # Dataset parameters
        self.num_classes = 10
        self.input_size = 32
        self.channels = 3
        
        # Standard CIFAR-10 normalization
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
    
    def _create_transforms(self):
        """Create various transform combinations"""
        
        # Basic transforms (no augmentation)
        self.transform_basic = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Standard training transforms
        self.transform_train_standard = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Advanced training transforms
        self.transform_train_advanced = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Test transforms
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Augmentation for OOD detection
        self.transform_ood_augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # No normalization (for raw features)
        self.transform_no_norm = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Grayscale transforms
        self.transform_grayscale = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def get_train_loader(self, transform_type='standard'):
        """
        Get training data loader
        
        Args:
            transform_type (str): Type of transforms to use
                Options: 'basic', 'standard', 'advanced', 'ood_augment', 'no_norm'
        
        Returns:
            DataLoader: Training data loader
        """
        transform_map = {
            'basic': self.transform_basic,
            'standard': self.transform_train_standard,
            'advanced': self.transform_train_advanced,
            'ood_augment': self.transform_ood_augment,
            'no_norm': self.transform_no_norm
        }
        
        if transform_type not in transform_map:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        dataset = CIFAR10Dataset(
            root=self.root_dir,
            train=True,
            transform=transform_map[transform_type],
            download=self.download
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def get_test_loader(self, transform_type='test'):
        """
        Get test data loader
        
        Args:
            transform_type (str): Type of transforms to use
                Options: 'test', 'basic', 'no_norm', 'grayscale'
        
        Returns:
            DataLoader: Test data loader
        """
        transform_map = {
            'test': self.transform_test,
            'basic': self.transform_basic,
            'no_norm': self.transform_no_norm,
            'grayscale': self.transform_grayscale
        }
        
        if transform_type not in transform_map:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        dataset = CIFAR10Dataset(
            root=self.root_dir,
            train=False,
            transform=transform_map[transform_type],
            download=self.download
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_subset_loader(self, indices, train=True, transform_type='test'):
        """
        Get data loader for a subset of the dataset
        
        Args:
            indices (list): List of indices to include
            train (bool): Whether to use training or test set
            transform_type (str): Type of transforms to use
        
        Returns:
            DataLoader: Subset data loader
        """
        # Create full dataset
        transform_map = {
            'test': self.transform_test,
            'basic': self.transform_basic,
            'no_norm': self.transform_no_norm,
            'standard': self.transform_train_standard
        }
        
        dataset = CIFAR10Dataset(
            root=self.root_dir,
            train=train,
            transform=transform_map.get(transform_type, self.transform_test),
            download=self.download
        )
        
        # Create subset
        subset = torch.utils.data.Subset(dataset, indices)
        
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_class_loader(self, class_idx, num_samples=None, train=True):
        """
        Get data loader for a specific class
        
        Args:
            class_idx (int): Class index
            num_samples (int, optional): Number of samples to include
            train (bool): Whether to use training or test set
        
        Returns:
            DataLoader: Class-specific data loader
        """
        dataset = CIFAR10Dataset(
            root=self.root_dir,
            train=train,
            transform=self.transform_test,
            download=self.download
        )
        
        class_indices = dataset.get_class_samples(class_idx, num_samples)
        
        subset = torch.utils.data.Subset(dataset, class_indices)
        
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_balanced_loader(self, samples_per_class=100, train=True):
        """
        Get balanced data loader with equal samples from each class
        
        Args:
            samples_per_class (int): Number of samples per class
            train (bool): Whether to use training or test set
        
        Returns:
            DataLoader: Balanced data loader
        """
        dataset = CIFAR10Dataset(
            root=self.root_dir,
            train=train,
            transform=self.transform_test,
            download=self.download
        )
        
        all_indices = []
        for class_idx in range(self.num_classes):
            class_indices = dataset.get_class_samples(class_idx, samples_per_class)
            all_indices.extend(class_indices)
        
        subset = torch.utils.data.Subset(dataset, all_indices)
        
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_dataset_info(self):
        """Get dataset information"""
        train_dataset = CIFAR10Dataset(
            root=self.root_dir,
            train=True,
            transform=None,
            download=self.download
        )
        
        test_dataset = CIFAR10Dataset(
            root=self.root_dir,
            train=False,
            transform=None,
            download=self.download
        )
        
        info = {
            'name': 'CIFAR-10',
            'num_classes': self.num_classes,
            'class_names': train_dataset.class_names,
            'input_size': self.input_size,
            'channels': self.channels,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'normalization': {
                'mean': self.mean,
                'std': self.std
            }
        }
        
        return info
    
    def get_class_distribution(self, train=True):
        """Get class distribution in the dataset"""
        dataset = CIFAR10Dataset(
            root=self.root_dir,
            train=train,
            transform=None,
            download=self.download
        )
        
        class_counts = {}
        for class_idx in range(self.num_classes):
            class_counts[dataset.class_names[class_idx]] = 0
        
        for target in dataset.targets:
            class_counts[dataset.class_names[target]] += 1
        
        return class_counts


def create_cifar10_loader(root_dir, batch_size=128, num_workers=4, download=True):
    """
    Factory function to create CIFAR-10 data loader
    
    Args:
        root_dir (str): Root directory for dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers
        download (bool): Whether to download dataset
    
    Returns:
        CIFAR10DataLoader: Configured data loader
    """
    return CIFAR10DataLoader(root_dir, batch_size, num_workers, download)


if __name__ == '__main__':
    # Test the data loader
    print("Testing CIFAR-10 DataLoader...")
    
    # Create data loader
    loader = CIFAR10DataLoader('./data', batch_size=64, num_workers=2)
    
    # Get dataset info
    info = loader.get_dataset_info()
    print(f"Dataset info: {info}")
    
    # Get class distribution
    train_dist = loader.get_class_distribution(train=True)
    print(f"Train distribution: {train_dist}")
    
    # Test data loaders
    train_loader = loader.get_train_loader('standard')
    test_loader = loader.get_test_loader('test')
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    # Test a batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"Target classes: {torch.unique(target)}")
        break
    
    print("CIFAR-10 DataLoader test completed successfully!")
