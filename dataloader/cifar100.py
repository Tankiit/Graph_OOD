"""
CIFAR-100 DataLoader for Graph-based Spectral OOD Detection

This module provides comprehensive data loading capabilities for CIFAR-100 dataset
with support for various augmentations and preprocessing techniques.

Key features:
1. Standard and advanced data augmentations
2. Support for different normalization schemes
3. Custom transforms for OOD detection
4. Efficient data loading with caching
5. Support for subset creation and sampling
6. Fine and coarse label support
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import logging


class CIFAR100Dataset(Dataset):
    """
    Custom CIFAR-100 dataset with enhanced functionality
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
        
        # Load CIFAR-100 dataset
        self.dataset = CIFAR100(root=root, train=train, download=download)
        self.data = self.dataset.data
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
        # Load fine and coarse labels
        self.fine_labels = self.dataset.targets
        self.coarse_labels = [self.dataset.targets[i] for i in range(len(self.dataset.targets))]
        
        # Load metadata for coarse labels
        if hasattr(self.dataset, 'coarse_labels'):
            self.coarse_labels = self.dataset.coarse_labels
        
        # CIFAR-100 fine class names (100 classes)
        self.fine_class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        
        # CIFAR-100 coarse class names (20 superclasses)
        self.coarse_class_names = [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
            'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
            'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
            'trees', 'vehicles_1', 'vehicles_2'
        ]
        
        # Create mapping from fine to coarse labels
        self._create_fine_to_coarse_mapping()
    
    def _create_fine_to_coarse_mapping(self):
        """Create mapping from fine labels to coarse labels"""
        # This mapping is based on CIFAR-100 dataset structure
        fine_to_coarse = [
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
            5, 19, 8, 8, 15, 13, 14, 17, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10,
            3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16,
            19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13, 14
        ]
        
        self.fine_to_coarse = fine_to_coarse
        self.coarse_labels = [fine_to_coarse[label] for label in self.fine_labels]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, fine_target, coarse_target) where targets are indices of classes
        """
        img, fine_target = self.data[idx], self.fine_labels[idx]
        coarse_target = self.coarse_labels[idx]
        
        # Convert to PIL Image
        img = torchvision.transforms.ToPILImage()(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, fine_target, coarse_target
    
    def get_class_samples(self, class_idx, num_samples=None, label_type='fine'):
        """
        Get samples from a specific class
        
        Args:
            class_idx (int): Class index
            num_samples (int, optional): Number of samples to return
            label_type (str): 'fine' or 'coarse' labels
        
        Returns:
            list: List of indices for the specified class
        """
        if label_type == 'fine':
            target_labels = self.fine_labels
        else:
            target_labels = self.coarse_labels
        
        class_indices = [i for i, target in enumerate(target_labels) if target == class_idx]
        
        if num_samples is not None:
            np.random.seed(42)  # For reproducibility
            class_indices = np.random.choice(class_indices, 
                                           min(num_samples, len(class_indices)), 
                                           replace=False).tolist()
        
        return class_indices
    
    def get_coarse_class_samples(self, coarse_class_idx, num_samples=None):
        """Get samples from a specific coarse class"""
        return self.get_class_samples(coarse_class_idx, num_samples, label_type='coarse')


class CIFAR100DataLoader:
    """
    Comprehensive CIFAR-100 data loader with multiple augmentation strategies
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
        self.num_fine_classes = 100
        self.num_coarse_classes = 20
        self.input_size = 32
        self.channels = 3
        
        # Standard CIFAR-100 normalization
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
    
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
        
        dataset = CIFAR100Dataset(
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
        
        dataset = CIFAR100Dataset(
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
        
        dataset = CIFAR100Dataset(
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
    
    def get_fine_class_loader(self, class_idx, num_samples=None, train=True):
        """
        Get data loader for a specific fine class
        
        Args:
            class_idx (int): Fine class index
            num_samples (int, optional): Number of samples to include
            train (bool): Whether to use training or test set
        
        Returns:
            DataLoader: Class-specific data loader
        """
        dataset = CIFAR100Dataset(
            root=self.root_dir,
            train=train,
            transform=self.transform_test,
            download=self.download
        )
        
        class_indices = dataset.get_class_samples(class_idx, num_samples, label_type='fine')
        
        subset = torch.utils.data.Subset(dataset, class_indices)
        
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_coarse_class_loader(self, coarse_class_idx, num_samples=None, train=True):
        """
        Get data loader for a specific coarse class
        
        Args:
            coarse_class_idx (int): Coarse class index
            num_samples (int, optional): Number of samples to include
            train (bool): Whether to use training or test set
        
        Returns:
            DataLoader: Coarse class-specific data loader
        """
        dataset = CIFAR100Dataset(
            root=self.root_dir,
            train=train,
            transform=self.transform_test,
            download=self.download
        )
        
        class_indices = dataset.get_coarse_class_samples(coarse_class_idx, num_samples)
        
        subset = torch.utils.data.Subset(dataset, class_indices)
        
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_balanced_loader(self, samples_per_class=50, train=True, label_type='fine'):
        """
        Get balanced data loader with equal samples from each class
        
        Args:
            samples_per_class (int): Number of samples per class
            train (bool): Whether to use training or test set
            label_type (str): 'fine' or 'coarse' labels
        
        Returns:
            DataLoader: Balanced data loader
        """
        dataset = CIFAR100Dataset(
            root=self.root_dir,
            train=train,
            transform=self.transform_test,
            download=self.download
        )
        
        num_classes = self.num_fine_classes if label_type == 'fine' else self.num_coarse_classes
        
        all_indices = []
        for class_idx in range(num_classes):
            class_indices = dataset.get_class_samples(class_idx, samples_per_class, label_type)
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
        train_dataset = CIFAR100Dataset(
            root=self.root_dir,
            train=True,
            transform=None,
            download=self.download
        )
        
        test_dataset = CIFAR100Dataset(
            root=self.root_dir,
            train=False,
            transform=None,
            download=self.download
        )
        
        info = {
            'name': 'CIFAR-100',
            'num_fine_classes': self.num_fine_classes,
            'num_coarse_classes': self.num_coarse_classes,
            'fine_class_names': train_dataset.fine_class_names,
            'coarse_class_names': train_dataset.coarse_class_names,
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
    
    def get_class_distribution(self, train=True, label_type='fine'):
        """Get class distribution in the dataset"""
        dataset = CIFAR100Dataset(
            root=self.root_dir,
            train=train,
            transform=None,
            download=self.download
        )
        
        if label_type == 'fine':
            num_classes = self.num_fine_classes
            class_names = dataset.fine_class_names
            labels = dataset.fine_labels
        else:
            num_classes = self.num_coarse_classes
            class_names = dataset.coarse_class_names
            labels = dataset.coarse_labels
        
        class_counts = {}
        for class_idx in range(num_classes):
            class_counts[class_names[class_idx]] = 0
        
        for label in labels:
            class_counts[class_names[label]] += 1
        
        return class_counts


def create_cifar100_loader(root_dir, batch_size=128, num_workers=4, download=True):
    """
    Factory function to create CIFAR-100 data loader
    
    Args:
        root_dir (str): Root directory for dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers
        download (bool): Whether to download dataset
    
    Returns:
        CIFAR100DataLoader: Configured data loader
    """
    return CIFAR100DataLoader(root_dir, batch_size, num_workers, download)


if __name__ == '__main__':
    # Test the data loader
    print("Testing CIFAR-100 DataLoader...")
    
    # Create data loader
    loader = CIFAR100DataLoader('./data', batch_size=64, num_workers=2)
    
    # Get dataset info
    info = loader.get_dataset_info()
    print(f"Dataset info: {info['name']}, {info['num_fine_classes']} fine classes, {info['num_coarse_classes']} coarse classes")
    
    # Get class distribution
    train_dist_fine = loader.get_class_distribution(train=True, label_type='fine')
    train_dist_coarse = loader.get_class_distribution(train=True, label_type='coarse')
    
    print(f"Fine classes: {len(train_dist_fine)}")
    print(f"Coarse classes: {len(train_dist_coarse)}")
    
    # Test data loaders
    train_loader = loader.get_train_loader('standard')
    test_loader = loader.get_test_loader('test')
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    # Test a batch
    for batch_idx, (data, fine_target, coarse_target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}")
        print(f"Fine target shape: {fine_target.shape}, Coarse target shape: {coarse_target.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"Fine target classes: {torch.unique(fine_target)}")
        print(f"Coarse target classes: {torch.unique(coarse_target)}")
        break
    
    print("CIFAR-100 DataLoader test completed successfully!")
