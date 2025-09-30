"""
Feature Caching System for Spectral OOD Detection
Avoids recomputing expensive feature extractions across experiments
"""

import os
import pickle
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

class FeatureCache:
    """
    Intelligent caching system for extracted features
    Supports different cache strategies and automatic invalidation
    """
    
    def __init__(self, cache_dir: str = './cache', max_cache_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        # Cache metadata
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_cache_key(self, dataset_name: str, architecture: str, 
                          max_samples: Optional[int] = None, 
                          additional_params: Optional[Dict] = None) -> str:
        """Generate unique cache key for feature extraction"""
        key_components = {
            'dataset': dataset_name,
            'architecture': architecture,
            'max_samples': max_samples,
            'params': additional_params or {}
        }
        
        # Create deterministic hash
        key_str = json.dumps(key_components, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        for file_path in self.cache_dir.glob("*.pkl"):
            total_size += file_path.stat().st_size
        return total_size
    
    def _cleanup_cache(self):
        """Remove old cache entries if size exceeds limit"""
        current_size = self._get_cache_size()
        
        if current_size <= self.max_cache_size:
            return
            
        self.logger.info(f"Cache size ({current_size / 1024**3:.2f} GB) exceeds limit. Cleaning up...")
        
        # Sort by last access time
        cache_files = []
        for cache_key, metadata in self.metadata.items():
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_files.append((cache_key, metadata.get('last_access', 0), cache_path))
        
        # Sort by last access time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove oldest files until under size limit
        for cache_key, _, cache_path in cache_files:
            if self._get_cache_size() <= self.max_cache_size * 0.8:  # 80% of limit
                break
                
            cache_path.unlink()
            if cache_key in self.metadata:
                del self.metadata[cache_key]
            
            self.logger.info(f"Removed cache entry: {cache_key}")
        
        self._save_metadata()
    
    def cache_features(self, features: np.ndarray, labels: np.ndarray,
                      dataset_name: str, architecture: str,
                      max_samples: Optional[int] = None,
                      additional_params: Optional[Dict] = None) -> str:
        """Cache extracted features"""
        
        cache_key = self._generate_cache_key(dataset_name, architecture, 
                                           max_samples, additional_params)
        cache_path = self._get_cache_path(cache_key)
        
        # Cache data
        cache_data = {
            'features': features,
            'labels': labels,
            'dataset_name': dataset_name,
            'architecture': architecture,
            'max_samples': max_samples,
            'additional_params': additional_params,
            'feature_shape': features.shape,
            'n_samples': len(features)
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            import time
            self.metadata[cache_key] = {
                'dataset_name': dataset_name,
                'architecture': architecture,
                'max_samples': max_samples,
                'feature_shape': list(features.shape),
                'n_samples': len(features),
                'cache_time': time.time(),
                'last_access': time.time(),
                'file_size': cache_path.stat().st_size
            }
            
            self._save_metadata()
            self._cleanup_cache()
            
            self.logger.info(f"Cached features for {dataset_name}/{architecture}: {cache_key}")
            return cache_key
            
        except Exception as e:
            self.logger.error(f"Failed to cache features: {e}")
            if cache_path.exists():
                cache_path.unlink()
            return None
    
    def load_features(self, dataset_name: str, architecture: str,
                     max_samples: Optional[int] = None,
                     additional_params: Optional[Dict] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load cached features if available"""
        
        cache_key = self._generate_cache_key(dataset_name, architecture,
                                           max_samples, additional_params)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Update last access time
            import time
            if cache_key in self.metadata:
                self.metadata[cache_key]['last_access'] = time.time()
                self._save_metadata()
            
            features = cache_data['features']
            labels = cache_data['labels']
            
            self.logger.info(f"Loaded cached features for {dataset_name}/{architecture}: {features.shape}")
            return features, labels
            
        except Exception as e:
            self.logger.error(f"Failed to load cached features: {e}")
            # Remove corrupted cache file
            if cache_path.exists():
                cache_path.unlink()
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()
            return None
    
    def is_cached(self, dataset_name: str, architecture: str,
                  max_samples: Optional[int] = None,
                  additional_params: Optional[Dict] = None) -> bool:
        """Check if features are cached"""
        cache_key = self._generate_cache_key(dataset_name, architecture,
                                           max_samples, additional_params)
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists() and cache_key in self.metadata
    
    def clear_cache(self, dataset_name: Optional[str] = None, 
                   architecture: Optional[str] = None):
        """Clear cache entries (optionally filtered)"""
        
        keys_to_remove = []
        
        for cache_key, metadata in self.metadata.items():
            should_remove = True
            
            if dataset_name and metadata.get('dataset_name') != dataset_name:
                should_remove = False
            if architecture and metadata.get('architecture') != architecture:
                should_remove = False
                
            if should_remove:
                keys_to_remove.append(cache_key)
        
        for cache_key in keys_to_remove:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
            if cache_key in self.metadata:
                del self.metadata[cache_key]
        
        self._save_metadata()
        self.logger.info(f"Cleared {len(keys_to_remove)} cache entries")
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        total_size = self._get_cache_size()
        
        return {
            'total_entries': len(self.metadata),
            'total_size_gb': total_size / (1024**3),
            'max_size_gb': self.max_cache_size / (1024**3),
            'cache_dir': str(self.cache_dir),
            'entries': list(self.metadata.keys())
        }
    
    def print_cache_info(self):
        """Print cache information"""
        info = self.get_cache_info()
        
        print(f"\n{'='*60}")
        print(f"FEATURE CACHE INFORMATION")
        print(f"{'='*60}")
        print(f"Cache Directory: {info['cache_dir']}")
        print(f"Total Entries: {info['total_entries']}")
        print(f"Total Size: {info['total_size_gb']:.2f} GB / {info['max_size_gb']:.1f} GB")
        print(f"Cache Utilization: {info['total_size_gb']/info['max_size_gb']*100:.1f}%")
        
        if self.metadata:
            print(f"\nCached Datasets/Architectures:")
            for cache_key, metadata in self.metadata.items():
                dataset = metadata.get('dataset_name', 'unknown')
                arch = metadata.get('architecture', 'unknown')
                shape = metadata.get('feature_shape', [])
                n_samples = metadata.get('n_samples', 0)
                size_mb = metadata.get('file_size', 0) / (1024**2)
                
                print(f"  {dataset}/{arch}: {n_samples} samples, "
                      f"shape={shape}, size={size_mb:.1f}MB")


class CachedFeatureExtractor:
    """
    Feature extractor with intelligent caching
    Automatically caches and reuses extracted features
    """
    
    def __init__(self, cache_dir: str = './cache', max_cache_size_gb: float = 10.0):
        self.cache = FeatureCache(cache_dir, max_cache_size_gb)
        self.logger = logging.getLogger(__name__)
        
    def extract_features_with_cache(self, dataloader, dataset_name: str, 
                                   architecture: str, max_samples: Optional[int] = None,
                                   force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features with caching
        
        Args:
            dataloader: PyTorch DataLoader
            dataset_name: Name of dataset for caching
            architecture: Architecture name for caching
            max_samples: Maximum samples to extract
            force_recompute: Force recomputation even if cached
            
        Returns:
            Tuple of (features, labels)
        """
        
        # Check if features are cached
        if not force_recompute and self.cache.is_cached(dataset_name, architecture, max_samples):
            self.logger.info(f"Loading cached features for {dataset_name}/{architecture}")
            cached_result = self.cache.load_features(dataset_name, architecture, max_samples)
            if cached_result is not None:
                return cached_result
        
        # Extract features if not cached or cache failed
        self.logger.info(f"Extracting features for {dataset_name}/{architecture}")
        
        # Import here to avoid circular imports
        from advanced_spectral_vision import AdvancedFeatureExtractor
        
        extractor = AdvancedFeatureExtractor(architecture=architecture)
        features, labels = extractor.extract_features(dataloader, max_samples=max_samples)
        
        # Cache the extracted features
        cache_key = self.cache.cache_features(
            features, labels, dataset_name, architecture, max_samples
        )
        
        if cache_key:
            self.logger.info(f"Features cached with key: {cache_key}")
        
        return features, labels
    
    def batch_extract_features(self, configurations: List[Dict], 
                             data_loaders: Dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract features for multiple configurations in batch
        
        Args:
            configurations: List of dicts with 'dataset', 'architecture', 'max_samples'
            data_loaders: Dict mapping dataset names to data loaders
            
        Returns:
            Dict mapping config keys to (features, labels) tuples
        """
        
        results = {}
        
        for config in configurations:
            dataset_name = config['dataset']
            architecture = config['architecture']
            max_samples = config.get('max_samples')
            
            if dataset_name not in data_loaders:
                self.logger.warning(f"No data loader for dataset: {dataset_name}")
                continue
            
            config_key = f"{dataset_name}_{architecture}_{max_samples}"
            
            try:
                features, labels = self.extract_features_with_cache(
                    data_loaders[dataset_name],
                    dataset_name,
                    architecture,
                    max_samples
                )
                
                results[config_key] = (features, labels)
                
            except Exception as e:
                self.logger.error(f"Failed to extract features for {config_key}: {e}")
                continue
        
        return results
    
    def get_cache_info(self):
        """Get cache information"""
        return self.cache.get_cache_info()
    
    def print_cache_info(self):
        """Print cache information"""
        self.cache.print_cache_info()
    
    def clear_cache(self, dataset_name: Optional[str] = None, 
                   architecture: Optional[str] = None):
        """Clear cache"""
        self.cache.clear_cache(dataset_name, architecture)