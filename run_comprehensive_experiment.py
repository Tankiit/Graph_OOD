#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for Spectral OOD Detection
Orchestrates large-scale evaluation across all datasets and architectures
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from spectral_ood_vision import VisionOODEvaluator, VisionDatasetLoader, ImageSpectralOODDetector
from advanced_spectral_vision import ComprehensiveVisionOODEvaluator, HybridSpectralOODDetector, AdvancedFeatureExtractor

class ExperimentOrchestrator:
    """
    Orchestrates comprehensive experiments across all configurations
    """
    
    def __init__(self, 
                 data_dir: str = './data',
                 results_dir: str = './results',
                 cache_dir: str = './cache',
                 max_samples_per_dataset: int = 2000):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.cache_dir = cache_dir
        self.max_samples = max_samples_per_dataset
        
        # Create results directory
        Path(self.results_dir).mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.results_dir}/experiment.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Experiment configurations
        self.configurations = self._define_experiment_configurations()
        
    def _define_experiment_configurations(self) -> List[Dict]:
        """Define all experimental configurations"""
        
        # Dataset configurations
        id_datasets = ['cifar10', 'cifar100', 'svhn', 'tiny_imagenet']
        ood_datasets = {
            'cifar10': ['cifar100', 'tiny_imagenet', 'svhn', 'noise', 'texture'],
            'cifar100': ['cifar10', 'tiny_imagenet', 'svhn', 'noise', 'texture'], 
            'tiny_imagenet': ['cifar10', 'cifar100', 'svhn', 'noise', 'texture'], 
            'svhn': ['cifar10', 'cifar100', 'tiny_imagenet', 'noise', 'texture']
        }
        
        # Architecture configurations
        architectures = {
            'lightweight': ['resnet18', 'efficientnet_b0'],
            'standard': ['resnet50', 'vgg16', 'densenet121'],
            'heavy': ['resnet101', 'efficientnet_b4'],
            'transformer': ['vit_base', 'swin_base']
        }
        
        # Method configurations
        methods = {
            'basic_spectral': {
                'class': 'ImageSpectralOODDetector',
                'params': {'method': 'spectral_gap', 'pca_dim': 256}
            },
            'multiscale_spectral': {
                'class': 'ImageSpectralOODDetector', 
                'params': {'method': 'multiscale', 'pca_dim': 256}
            },
            'unified_spectral': {
                'class': 'ImageSpectralOODDetector',
                'params': {'method': 'unified', 'pca_dim': 512}
            },
            'hybrid_advanced': {
                'class': 'HybridSpectralOODDetector',
                'params': {
                    'embedding_dim': 64,
                    'pca_components': 512,
                    'adaptive_k': True,
                    'ensemble_methods': ['spectral_gap', 'heat_kernel']
                }
            },
            'hybrid_full': {
                'class': 'HybridSpectralOODDetector',
                'params': {
                    'embedding_dim': 64,
                    'pca_components': 512, 
                    'adaptive_k': True,
                    'ensemble_methods': ['spectral_gap', 'heat_kernel', 'topology']
                }
            }
        }
        
        # Generate all combinations
        configurations = []
        config_id = 0
        
        for id_dataset in id_datasets:
            for ood_dataset in ood_datasets[id_dataset]:
                for arch_category, archs in architectures.items():
                    for arch in archs:
                        for method_name, method_config in methods.items():
                            configurations.append({
                                'config_id': config_id,
                                'id_dataset': id_dataset,
                                'ood_dataset': ood_dataset,
                                'architecture': arch,
                                'architecture_category': arch_category,
                                'method_name': method_name,
                                'method_config': method_config,
                                'priority': self._get_priority(id_dataset, ood_dataset, arch, method_name)
                            })
                            config_id += 1
        
        # Sort by priority (high priority first)
        configurations.sort(key=lambda x: x['priority'], reverse=True)
        
        self.logger.info(f"Generated {len(configurations)} experimental configurations")
        return configurations
    
    def _get_priority(self, id_dataset: str, ood_dataset: str, arch: str, method: str) -> int:
        """Assign priority to configurations for execution order"""
        priority = 0
        
        # Higher priority for important dataset combinations
        important_combos = [
            ('cifar10', 'cifar100'), ('cifar10', 'svhn'),
            ('cifar100', 'cifar10'), ('svhn', 'cifar10')
        ]
        if (id_dataset, ood_dataset) in important_combos:
            priority += 10
            
        # Higher priority for standard architectures
        if arch in ['resnet50', 'resnet18']:
            priority += 5
        
        # Higher priority for advanced methods
        if 'hybrid' in method:
            priority += 3
        elif 'unified' in method:
            priority += 2
            
        return priority
    
    def run_single_experiment(self, config: Dict) -> Dict:
        """Run a single experimental configuration"""
        
        self.logger.info(f"Running experiment {config['config_id']}: "
                        f"{config['id_dataset']} vs {config['ood_dataset']} | "
                        f"{config['architecture']} | {config['method_name']}")
        
        try:
            start_time = time.time()
            
            # Initialize data loader
            loader = VisionDatasetLoader(self.data_dir, batch_size=32)
            
            # Load ID dataset
            if config['id_dataset'] == 'cifar10':
                id_loader = loader.get_cifar10(train=False)
            elif config['id_dataset'] == 'cifar100':
                id_loader = loader.get_cifar100(train=False) 
            elif config['id_dataset'] == 'svhn':
                id_loader = loader.get_svhn(split='test')
            elif config['id_dataset'] == 'tiny_imagenet':
                id_loader = loader.get_tiny_imagenet(train=False)
            else:
                raise ValueError(f"Unknown ID dataset: {config['id_dataset']}")
            
            # Load OOD dataset
            if config['ood_dataset'] == 'cifar10':
                ood_loader = loader.get_cifar10(train=False)
            elif config['ood_dataset'] == 'cifar100':
                ood_loader = loader.get_cifar100(train=False)
            elif config['ood_dataset'] == 'svhn':
                ood_loader = loader.get_svhn(split='test')
            elif config['ood_dataset'] == 'tiny_imagenet':
                ood_loader = loader.get_tiny_imagenet(train=False)
            elif config['ood_dataset'] == 'noise':
                ood_loader = loader.get_noise_ood(size=self.max_samples//2)
            elif config['ood_dataset'] == 'texture':
                ood_loader = loader.get_texture_ood()
            else:
                raise ValueError(f"Unknown OOD dataset: {config['ood_dataset']}")
            
                # Extract features with caching
            feature_extractor = AdvancedFeatureExtractor(architecture=config['architecture'])
            
            # Use cached features if available
            id_features, id_labels = feature_extractor.extract_features_with_cache(
                id_loader, config['id_dataset'], max_samples=self.max_samples,
                cache_dir=self.cache_dir
            )
            ood_features, ood_labels = feature_extractor.extract_features_with_cache(
                ood_loader, config['ood_dataset'], max_samples=self.max_samples//2,
                cache_dir=self.cache_dir
            )
            
            # Prepare train/test splits
            n_train = min(1000, len(id_features) // 2)
            n_test_id = min(500, len(id_features) - n_train)
            
            train_features = id_features[:n_train]
            test_id_features = id_features[n_train:n_train + n_test_id]
            
            # Combine test data
            test_features = np.vstack([test_id_features, ood_features])
            test_labels = np.concatenate([np.zeros(len(test_id_features)), np.ones(len(ood_features))])
            
            # Initialize detector based on method
            method_config = config['method_config']
            if method_config['class'] == 'ImageSpectralOODDetector':
                detector = ImageSpectralOODDetector(**method_config['params'])
            elif method_config['class'] == 'HybridSpectralOODDetector':
                detector = HybridSpectralOODDetector(**method_config['params'])
            else:
                raise ValueError(f"Unknown detector class: {method_config['class']}")
            
            # Fit detector
            detector.fit(train_features)
            
            # Predict scores
            scores = detector.predict_score(test_features)
            
            # Evaluate
            from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
            
            auc = roc_auc_score(test_labels, scores)
            ap = average_precision_score(test_labels, scores)
            
            # Additional metrics
            fpr, tpr, _ = roc_curve(test_labels, scores)
            precision, recall, _ = precision_recall_curve(test_labels, scores)
            
            # FPR at TPR=95%
            tpr95_idx = np.where(tpr >= 0.95)[0]
            fpr95 = fpr[tpr95_idx[0]] if len(tpr95_idx) > 0 else 1.0
            
            # AUROC at different FPR levels
            fpr80_idx = np.where(tpr >= 0.80)[0]
            fpr80 = fpr[fpr80_idx[0]] if len(fpr80_idx) > 0 else 1.0
            
            execution_time = time.time() - start_time
            
            # Compile results
            result = {
                **config,
                'auc': float(auc),
                'average_precision': float(ap),
                'fpr95': float(fpr95),
                'fpr80': float(fpr80),
                'n_train': n_train,
                'n_test_id': len(test_id_features),
                'n_test_ood': len(ood_features),
                'feature_dim_original': id_features.shape[1],
                'execution_time': execution_time,
                'success': True,
                'error_message': None
            }
            
            self.logger.info(f"Experiment {config['config_id']} completed successfully: "
                           f"AUC={auc:.4f}, AP={ap:.4f}, FPR95={fpr95:.4f}, "
                           f"Time={execution_time:.1f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment {config['config_id']} failed: {str(e)}")
            
            return {
                **config,
                'auc': None,
                'average_precision': None,
                'fpr95': None,
                'fpr80': None,
                'n_train': None,
                'n_test_id': None,
                'n_test_ood': None,
                'feature_dim_original': None,
                'execution_time': None,
                'success': False,
                'error_message': str(e)
            }
    
    def run_comprehensive_experiments(self, 
                                    max_experiments: int = None,
                                    architecture_filter: List[str] = None,
                                    method_filter: List[str] = None):
        """Run comprehensive experiments with optional filters"""
        
        # Apply filters
        filtered_configs = self.configurations.copy()
        
        if architecture_filter:
            filtered_configs = [c for c in filtered_configs if c['architecture'] in architecture_filter]
            
        if method_filter:
            filtered_configs = [c for c in filtered_configs if c['method_name'] in method_filter]
        
        if max_experiments:
            filtered_configs = filtered_configs[:max_experiments]
        
        self.logger.info(f"Running {len(filtered_configs)} experiments")
        
        # Run experiments
        results = []
        successful_experiments = 0
        
        for i, config in enumerate(filtered_configs):
            self.logger.info(f"Progress: {i+1}/{len(filtered_configs)}")
            
            result = self.run_single_experiment(config)
            results.append(result)
            
            if result['success']:
                successful_experiments += 1
            
            # Save intermediate results every 10 experiments
            if (i + 1) % 10 == 0:
                self._save_results(results, f"intermediate_results_{i+1}.json")
        
        # Save final results
        self._save_results(results, "comprehensive_results.json")
        
        self.logger.info(f"Experiments completed: {successful_experiments}/{len(filtered_configs)} successful")
        
        return results
    
    def _save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        filepath = Path(self.results_dir) / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {filepath}")
    
    def analyze_results(self, results_file: str = "comprehensive_results.json") -> pd.DataFrame:
        """Comprehensive analysis of experimental results"""
        
        # Load results
        results_path = Path(self.results_dir) / results_file
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Filter successful results
        successful_results = [r for r in results if r['success']]
        df = pd.DataFrame(successful_results)
        
        if df.empty:
            self.logger.warning("No successful results to analyze")
            return df
        
        self.logger.info(f"Analyzing {len(df)} successful experiments")
        
        # Generate comprehensive analysis
        self._generate_summary_statistics(df)
        self._generate_detailed_analysis(df)
        self._create_comprehensive_visualizations(df)
        
        return df
    
    def _generate_summary_statistics(self, df: pd.DataFrame):
        """Generate summary statistics"""
        
        print("\n" + "="*100)
        print("COMPREHENSIVE EXPERIMENTAL RESULTS SUMMARY")
        print("="*100)
        
        # Overall performance
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Total experiments: {len(df)}")
        print(f"Mean AUC: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
        print(f"Mean AP: {df['average_precision'].mean():.4f} ± {df['average_precision'].std():.4f}")
        print(f"Mean FPR95: {df['fpr95'].mean():.4f} ± {df['fpr95'].std():.4f}")
        print(f"Mean execution time: {df['execution_time'].mean():.1f}s ± {df['execution_time'].std():.1f}s")
        
        # Top performers
        print(f"\nTOP 10 CONFIGURATIONS BY AUC:")
        top10 = df.nlargest(10, 'auc')[['config_id', 'id_dataset', 'ood_dataset', 
                                        'architecture', 'method_name', 'auc', 'average_precision', 'fpr95']]
        print(top10.to_string(index=False))
        
        # Method comparison
        print(f"\nPERFORMANCE BY METHOD:")
        method_stats = df.groupby('method_name')[['auc', 'average_precision', 'fpr95', 'execution_time']].agg(['mean', 'std', 'count'])
        print(method_stats.round(4))
        
        # Architecture comparison
        print(f"\nPERFORMANCE BY ARCHITECTURE:")
        arch_stats = df.groupby('architecture')[['auc', 'average_precision', 'fpr95']].agg(['mean', 'std', 'count'])
        print(arch_stats.round(4))
    
    def _generate_detailed_analysis(self, df: pd.DataFrame):
        """Generate detailed statistical analysis"""
        
        print(f"\nDETAILED STATISTICAL ANALYSIS:")
        print("-" * 50)
        
        # Statistical tests between methods
        from scipy.stats import ttest_ind, f_oneway
        
        methods = df['method_name'].unique()
        print(f"\nPairwise t-tests between methods (AUC scores):")
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                scores1 = df[df['method_name'] == method1]['auc'].dropna()
                scores2 = df[df['method_name'] == method2]['auc'].dropna()
                if len(scores1) > 1 and len(scores2) > 1:
                    stat, p_val = ttest_ind(scores1, scores2)
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"{method1} vs {method2}: t={stat:.3f}, p={p_val:.4f} {significance}")
        
        # ANOVA test
        method_groups = [df[df['method_name'] == method]['auc'].dropna() for method in methods]
        method_groups = [group for group in method_groups if len(group) > 1]
        if len(method_groups) > 1:
            f_stat, p_val = f_oneway(*method_groups)
            print(f"\nANOVA test across methods: F={f_stat:.3f}, p={p_val:.4f}")
        
        # Dataset difficulty ranking
        print(f"\nDATASET COMBINATION DIFFICULTY (by mean AUC, ascending):")
        dataset_difficulty = df.groupby(['id_dataset', 'ood_dataset'])['auc'].mean().sort_values()
        for (id_ds, ood_ds), auc in dataset_difficulty.items():
            print(f"{id_ds} vs {ood_ds}: {auc:.4f}")
        
        # Architecture efficiency analysis
        print(f"\nARCHITECTURE EFFICIENCY (AUC per second):")
        efficiency = df.groupby('architecture').apply(lambda x: x['auc'].mean() / x['execution_time'].mean()).sort_values(ascending=False)
        for arch, eff in efficiency.items():
            print(f"{arch}: {eff:.6f}")
    
    def _create_comprehensive_visualizations(self, df: pd.DataFrame):
        """Create comprehensive visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. AUC distribution
        ax1 = fig.add_subplot(gs[0, 0])
        df['auc'].hist(bins=30, alpha=0.7, ax=ax1, color='skyblue', edgecolor='black')
        ax1.axvline(df['auc'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["auc"].mean():.3f}')
        ax1.set_xlabel('AUC Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('AUC Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Method comparison boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        df.boxplot(column='auc', by='method_name', ax=ax2)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('Performance by Method')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Architecture comparison
        ax3 = fig.add_subplot(gs[0, 2])
        arch_means = df.groupby('architecture')['auc'].mean().sort_values(ascending=True)
        ax3.barh(range(len(arch_means)), arch_means.values, color='lightgreen')
        ax3.set_yticks(range(len(arch_means)))
        ax3.set_yticklabels(arch_means.index)
        ax3.set_xlabel('Mean AUC')
        ax3.set_title('Performance by Architecture')
        ax3.grid(True, alpha=0.3)
        
        # 4. Dataset combination heatmap
        ax4 = fig.add_subplot(gs[0, 3])
        pivot_auc = df.pivot_table(values='auc', index='id_dataset', columns='ood_dataset', aggfunc='mean')
        sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
        ax4.set_title('AUC by Dataset Combination')
        
        # 5. Performance vs execution time scatter
        ax5 = fig.add_subplot(gs[1, 0])
        scatter = ax5.scatter(df['execution_time'], df['auc'], 
                            c=df['method_name'].astype('category').cat.codes, 
                            alpha=0.6, s=50, cmap='tab10')
        ax5.set_xlabel('Execution Time (seconds)')
        ax5.set_ylabel('AUC Score')
        ax5.set_title('Performance vs Execution Time')
        ax5.grid(True, alpha=0.3)
        
        # 6. AUC vs FPR95 scatter
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.scatter(df['auc'], df['fpr95'], alpha=0.6, s=50, color='orange')
        ax6.set_xlabel('AUC Score')
        ax6.set_ylabel('FPR95')
        ax6.set_title('AUC vs FPR95')
        ax6.grid(True, alpha=0.3)
        
        # 7. Method-Architecture performance heatmap
        ax7 = fig.add_subplot(gs[1, 2:])
        pivot_method_arch = df.pivot_table(values='auc', index='method_name', columns='architecture', aggfunc='mean')
        sns.heatmap(pivot_method_arch, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax7)
        ax7.set_title('Method vs Architecture Performance')
        
        # 8. Dataset difficulty analysis
        ax8 = fig.add_subplot(gs[2, :2])
        dataset_perf = df.groupby(['id_dataset', 'ood_dataset'])['auc'].agg(['mean', 'std']).reset_index()
        dataset_perf['combo'] = dataset_perf['id_dataset'] + ' vs ' + dataset_perf['ood_dataset']
        dataset_perf = dataset_perf.sort_values('mean')
        
        ax8.errorbar(range(len(dataset_perf)), dataset_perf['mean'], 
                    yerr=dataset_perf['std'], fmt='o-', capsize=5)
        ax8.set_xticks(range(len(dataset_perf)))
        ax8.set_xticklabels(dataset_perf['combo'], rotation=45, ha='right')
        ax8.set_ylabel('AUC Score')
        ax8.set_title('Dataset Combination Difficulty')
        ax8.grid(True, alpha=0.3)
        
        # 9. Correlation matrix
        ax9 = fig.add_subplot(gs[2, 2])
        corr_vars = ['auc', 'average_precision', 'fpr95', 'execution_time']
        corr_matrix = df[corr_vars].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9)
        ax9.set_title('Metric Correlations')
        
        # 10. Performance distribution by architecture category
        ax10 = fig.add_subplot(gs[2, 3])
        if 'architecture_category' in df.columns:
            df.boxplot(column='auc', by='architecture_category', ax=ax10)
            ax10.set_xlabel('Architecture Category')
            ax10.set_ylabel('AUC Score')
            ax10.set_title('Performance by Architecture Category')
        
        # 11-12. Additional detailed plots
        ax11 = fig.add_subplot(gs[3, :2])
        # Feature dimension impact
        ax11.scatter(df['feature_dim_original'], df['auc'], alpha=0.6, s=50, color='purple')
        ax11.set_xlabel('Original Feature Dimension')
        ax11.set_ylabel('AUC Score')
        ax11.set_title('Feature Dimensionality Impact')
        ax11.grid(True, alpha=0.3)
        
        ax12 = fig.add_subplot(gs[3, 2:])
        # Training set size impact
        ax12.scatter(df['n_train'], df['auc'], alpha=0.6, s=50, color='brown')
        ax12.set_xlabel('Training Set Size')
        ax12.set_ylabel('AUC Score')
        ax12.set_title('Training Set Size Impact')
        ax12.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Spectral OOD Detection Analysis', fontsize=16, fontweight='bold')
        
        # Save figure
        output_path = Path(self.results_dir) / 'comprehensive_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Comprehensive visualization saved to {output_path}")


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Run comprehensive spectral OOD detection experiments')
    parser.add_argument('--data_dir', default='./data', help='Data directory')
    parser.add_argument('--cache_dir', default='./cache', help='Cache directory for features')
    parser.add_argument('--results_dir', default='./results', help='Results directory')
    parser.add_argument('--max_experiments', type=int, help='Maximum number of experiments to run')
    parser.add_argument('--architectures', nargs='+', help='Filter by architectures')
    parser.add_argument('--methods', nargs='+', help='Filter by methods')
    parser.add_argument('--max_samples', type=int, default=2000, help='Maximum samples per dataset')
    parser.add_argument('--quick_demo', action='store_true', help='Run quick demo with limited configurations')
    parser.add_argument('--analysis_only', help='Analyze existing results file')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation of cached features')
    parser.add_argument('--clear_cache', action='store_true', help='Clear all cached features before running')
    parser.add_argument('--cache_info', action='store_true', help='Show cache information and exit')
    
    args = parser.parse_args()
    
    # Import caching system
    from feature_cache import CachedFeatureExtractor
    
    # Initialize cache system
    cache_system = CachedFeatureExtractor(cache_dir=args.cache_dir)
    
    # Handle cache operations
    if args.cache_info:
        cache_system.print_cache_info()
        return
        
    if args.clear_cache:
        print("\n🧹 Clearing cache...")
        cache_system.clear_cache()
        print("Cache cleared successfully!")
    
    # Initialize orchestrator
    orchestrator = ExperimentOrchestrator(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        max_samples_per_dataset=args.max_samples
    )
    
    if args.analysis_only:
        # Only run analysis
        orchestrator.analyze_results(args.analysis_only)
        return
    
    print("🚀 COMPREHENSIVE SPECTRAL OOD DETECTION EXPERIMENTS")
    print("="*80)
    print(f"Data Directory: {args.data_dir}")
    print(f"Cache Directory: {args.cache_dir}")
    print(f"Results Directory: {args.results_dir}")
    print(f"Total configurations available: {len(orchestrator.configurations)}")
    
    # Show cache info
    print("\n📊 Cache Information:")
    cache_system.print_cache_info()
    
    if args.quick_demo:
        print("🔬 Running quick demo with limited configurations...")
        results = orchestrator.run_comprehensive_experiments(
            max_experiments=5,
            architecture_filter=['resnet18'],
            method_filter=['hybrid_advanced']
        )
    else:
        print("🔥 Running comprehensive experiments...")
        results = orchestrator.run_comprehensive_experiments(
            max_experiments=args.max_experiments,
            architecture_filter=args.architectures,
            method_filter=args.methods
        )
    
    # Analyze results
    print("\n📊 Analyzing results...")
    df = orchestrator.analyze_results()
    
    print(f"\n✅ Experiments completed successfully!")
    print(f"📁 Results saved in: {args.results_dir}")
    print(f"📊 Analysis plots saved as: {args.results_dir}/comprehensive_analysis.png")
    
    # Summary statistics
    if not df.empty:
        successful_rate = len(df) / len(results) * 100
        best_config = df.loc[df['auc'].idxmax()]
        
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"Success rate: {successful_rate:.1f}%")
        print(f"Best configuration: {best_config['id_dataset']} vs {best_config['ood_dataset']} | "
              f"{best_config['architecture']} | {best_config['method_name']}")
        print(f"Best AUC: {best_config['auc']:.4f}")


if __name__ == "__main__":
    main()