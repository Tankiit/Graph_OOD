"""
download_checkpoints.py
Helper script to download pretrained WRN checkpoints and OOD datasets.

This script provides automated downloading for:
1. WRN-40-2 checkpoints from SAL repo
2. Textures (DTD) dataset
3. LSUN-C dataset

Usage:
    python download_checkpoints.py --all
    python download_checkpoints.py --wrn --dtd
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import urllib.request
import tarfile
import zipfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url, destination):
    """Download a file from URL to destination"""
    logger.info(f"Downloading {url}")

    try:
        # Create destination directory if it doesn't exist
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress bar
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print()  # New line after progress bar

        logger.info(f"✓ Downloaded to {destination}")
        return True
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract tar.gz or zip archive"""
    logger.info(f"Extracting {archive_path}")

    try:
        archive_path = Path(archive_path)
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)

        if archive_path.suffixes == ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            logger.error(f"Unknown archive format: {archive_path}")
            return False

        logger.info(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        return False


def download_wrn_checkpoints():
    """Download WRN-40-2 checkpoints"""
    logger.info("\n" + "="*60)
    logger.info("Downloading WRN Checkpoints")
    logger.info("="*60)

    checkpoints_dir = Path('./checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    # NOTE: These URLs are placeholders. You need to find the actual URLs
    # for WRN-40-2 checkpoints trained on CIFAR-10 and CIFAR-100.

    logger.info("\n⚠️  IMPORTANT:")
    logger.info("Pretrained WRN-40-2 checkpoints need to be obtained manually.")
    logger.info("\nOptions:")
    logger.info("1. Clone SAL repo: git clone https://github.com/deeplearning-wisc/sal.git")
    logger.info("2. Check if they provide pretrained weights")
    logger.info("3. Train WRN-40-2 yourself using standard CIFAR training procedures")
    logger.info("\nExpected paths:")
    logger.info("  - ./checkpoints/wrn40_2_cifar10.pt")
    logger.info("  - ./checkpoints/wrn40_2_cifar100.pt")

    # Try to find checkpoints in SAL repo if it exists
    sal_repo = Path('./sal')
    if sal_repo.exists():
        logger.info("\nSearching for checkpoints in SAL repo...")
        # Add search logic here

    return False  # Return False since manual download is required


def download_dtd():
    """Download Describable Textures Dataset (DTD)"""
    logger.info("\n" + "="*60)
    logger.info("Downloading Textures (DTD) Dataset")
    logger.info("="*60)

    data_dir = Path('/Users/tanmoy/research/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    dtd_url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/dtd-r1.0.1.tar.gz"
    dtd_archive = data_dir / 'dtd-r1.0.1.tar.gz'

    if (data_dir / 'dtd' / 'images').exists():
        logger.info("✓ DTD dataset already exists")
        return True

    # Download
    if not dtd_archive.exists():
        if not download_file(dtd_url, dtd_archive):
            return False

    # Extract
    if extract_archive(dtd_archive, data_dir):
        logger.info("✓ DTD dataset ready")
        return True

    return False


def download_lsun_c():
    """Download LSUN-C dataset"""
    logger.info("\n" + "="*60)
    logger.info("Downloading LSUN-C Dataset")
    logger.info("="*60)

    data_dir = Path('/Users/tanmoy/research/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n⚠️  IMPORTANT:")
    logger.info("LSUN-C dataset needs to be obtained manually.")
    logger.info("\nInstructions:")
    logger.info("1. Visit: https://github.com/yuval-alaluf/supervised-modeling-ood")
    logger.info("2. Follow their instructions to download LSUN-C")
    logger.info("3. Extract to: /Users/tanmoy/research/data/lsun_c/")

    # NOTE: LSUN-C might have specific download instructions
    # or require using a specific tool/script

    return False  # Return False since manual download is required


def main():
    parser = argparse.ArgumentParser(description='Download checkpoints and datasets')

    parser.add_argument('--wrn', action='store_true',
                       help='Download WRN checkpoints (manual)')
    parser.add_argument('--dtd', action='store_true',
                       help='Download Textures (DTD) dataset')
    parser.add_argument('--lsun', action='store_true',
                       help='Download LSUN-C dataset (manual)')
    parser.add_argument('--all', action='store_true',
                       help='Download all datasets')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip downloads if files already exist')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    logger.info("Download helper for UAI OOD Detection Experiments")
    logger.info("="*60)

    success = True

    if args.all or args.dtd:
        if not download_dtd():
            success = False

    if args.all or args.lsun:
        if not download_lsun_c():
            success = False

    if args.all or args.wrn:
        if not download_wrn_checkpoints():
            success = False

    logger.info("\n" + "="*60)
    if success:
        logger.info("Downloads complete!")
    else:
        logger.warning("Some downloads require manual intervention.")
        logger.warning("Please follow the instructions above.")
    logger.info("="*60)


if __name__ == '__main__':
    main()
