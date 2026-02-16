"""
run_experiments.py
Quick-start script to run the complete UAI OOD detection experiment pipeline.

This script orchestrates the entire workflow:
1. Feature extraction (WRN, CLIP, DINOv2)
2. Metrics computation (γ, η, λ₂)
3. OOD detection evaluation
4. Plot generation

Usage:
    python run_experiments.py --extract --evaluate --plot
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")

    issues = []

    # Check if WRN checkpoints exist
    wrn_cifar10 = Path('./checkpoints/wrn40_2_cifar10.pt')
    wrn_cifar100 = Path('./checkpoints/wrn40_2_cifar100.pt')

    if not wrn_cifar10.exists():
        issues.append("WRN CIFAR-10 checkpoint not found at checkpoints/wrn40_2_cifar10.pt")
    if not wrn_cifar100.exists():
        issues.append("WRN CIFAR-100 checkpoint not found at checkpoints/wrn40_2_cifar100.pt")

    # Check if OOD datasets exist
    dtd_path = Path('/Users/tanmoy/research/data/dtd/images')
    lsun_path = Path('/Users/tanmoy/research/data/lsun_c')

    if not dtd_path.exists():
        issues.append("Textures (DTD) dataset not found at /Users/tanmoy/research/data/dtd/images")

    if not lsun_path.exists():
        issues.append("LSUN-C dataset not found at /Users/tanmoy/research/data/lsun_c")

    # Check if required packages are installed
    try:
        import torch
        import torchvision
        import sklearn
        import open_clip
    except ImportError as e:
        issues.append(f"Missing required package: {e}")

    if issues:
        logger.warning("\n" + "="*60)
        logger.warning("PREREQUISITE ISSUES FOUND:")
        logger.warning("="*60)
        for i, issue in enumerate(issues, 1):
            logger.warning(f"{i}. {issue}")
        logger.warning("="*60)
        logger.warning("Some features may not work correctly.")
        logger.warning("Continue anyway? (y/n)")

        response = input().strip().lower()
        if response != 'y':
            logger.error("Aborting. Please fix the issues above and try again.")
            return False
    else:
        logger.info("✓ All prerequisites met!")

    return True


def run_feature_extraction():
    """Run feature extraction pipeline"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Feature Extraction")
    logger.info("="*60)

    try:
        from extract_features import main as extract_main
        extract_main()
        logger.info("✓ Feature extraction complete!")
        return True
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation():
    """Run metrics computation and OOD evaluation"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Metrics Computation and OOD Evaluation")
    logger.info("="*60)

    try:
        from compute_metrics_and_eval import main as eval_main
        eval_main()
        logger.info("✓ Evaluation complete!")
        return True
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_plotting():
    """Run plotting pipeline"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Plot Generation")
    logger.info("="*60)

    try:
        # Plotting is already done in eval_main, but we can regenerate plots
        from compute_metrics_and_eval import (
            plot_gamma_vs_auroc,
            plot_eta_vs_auroc,
            plot_lambda2_vs_auroc
        )
        import pandas as pd

        results_path = Path('./results/evaluation_results.csv')
        if not results_path.exists():
            logger.error(f"Results not found at {results_path}. Run evaluation first.")
            return False

        results_df = pd.read_csv(results_path)

        plot_gamma_vs_auroc(results_df)
        plot_eta_vs_auroc(results_df)
        plot_lambda2_vs_auroc(results_df)

        logger.info("✓ Plot generation complete!")
        return True
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """Print experiment summary"""
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)

    # Check results
    results_path = Path('./results/evaluation_results.csv')
    plots_path = Path('./plots/gamma_vs_auroc.png')

    if results_path.exists():
        logger.info(f"✓ Results saved to: {results_path}")

        import pandas as pd
        df = pd.read_csv(results_path)

        logger.info(f"\nTotal experiments run: {len(df)}")
        logger.info("\nBreakdown by architecture:")
        for arch in ['wrn', 'clip', 'dinov2']:
            count = len(df[df['architecture'] == arch])
            logger.info(f"  {arch.upper()}: {count} experiments")

        logger.info("\nAverage AUROCs:")
        for arch in ['wrn', 'clip', 'dinov2']:
            arch_data = df[df['architecture'] == arch]
            if len(arch_data) > 0:
                logger.info(f"\n  {arch.upper()}:")
                logger.info(f"    Energy:      {arch_data['auroc_energy'].mean():.4f}")
                logger.info(f"    Mahalanobis: {arch_data['auroc_mahalanobis'].mean():.4f}")
                logger.info(f"    k-NN:        {arch_data['auroc_knn'].mean():.4f}")
                logger.info(f"    Spectral:    {arch_data['auroc_spectral'].mean():.4f}")
    else:
        logger.warning("Results not found. Run evaluation first.")

    if plots_path.exists():
        logger.info(f"\n✓ Plots saved to: {plots_path.parent}")
    else:
        logger.warning("Plots not found. Run plotting first.")

    logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Run UAI OOD Detection Experiments')

    parser.add_argument('--extract', action='store_true',
                       help='Run feature extraction')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run metrics computation and OOD evaluation')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--all', action='store_true',
                       help='Run all steps (extract, evaluate, plot)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip prerequisite checks')
    parser.add_argument('--summary', action='store_true',
                       help='Print experiment summary')

    args = parser.parse_args()

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            return

    # Run requested steps
    success = True

    if args.all:
        success = run_feature_extraction() and success
        success = run_evaluation() and success
        success = run_plotting() and success
    else:
        if args.extract:
            success = run_feature_extraction() and success
        if args.evaluate:
            success = run_evaluation() and success
        if args.plot:
            success = run_plotting() and success

    # Print summary
    if args.summary or args.all:
        print_summary()

    if success:
        logger.info("\n" + "="*60)
        logger.info("ALL TASKS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
    else:
        logger.error("\nSome tasks failed. Check the logs above for details.")


if __name__ == '__main__':
    main()
