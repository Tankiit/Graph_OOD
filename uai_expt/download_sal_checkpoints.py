"""
download_sal_checkpoints.py
Download pretrained WRN checkpoints from the SAL repository.

This script automatically downloads:
- cifar10_wrn_pretrained_epoch_99.pt
- cifar100_wrn_pretrained_epoch_99.pt

Usage:
    python download_sal_checkpoints.py
"""

import urllib.request
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url, destination):
    """Download a file from URL to destination with progress bar"""
    logger.info(f"Downloading {url}")

    try:
        # Create destination directory if it doesn't exist
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists
        if destination.exists():
            logger.info(f"✓ File already exists: {destination}")
            return True

        # Download with progress bar
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print()  # New line after progress bar

        logger.info(f"✓ Downloaded to {destination}")
        return True
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return False


def main():
    """Download SAL pretrained checkpoints"""

    logger.info("="*60)
    logger.info("Downloading SAL Pretrained Checkpoints")
    logger.info("="*60)

    # Base URL for raw GitHub content
    base_url = "https://github.com/deeplearning-wisc/sal/raw/main/pretrained"

    # Checkpoints directory
    checkpoints_dir = Path('./checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    # Files to download
    checkpoints = {
        'cifar10': {
            'url': f"{base_url}/cifar10_wrn_pretrained_epoch_99.pt",
            'dest': checkpoints_dir / 'wrn40_2_cifar10.pt'
        },
        'cifar100': {
            'url': f"{base_url}/cifar100_wrn_pretrained_epoch_99.pt",
            'dest': checkpoints_dir / 'wrn40_2_cifar100.pt'
        }
    }

    # Download each checkpoint
    success = True
    for name, info in checkpoints.items():
        logger.info(f"\nDownloading {name} checkpoint...")
        if not download_file(info['url'], info['dest']):
            success = False

    logger.info("\n" + "="*60)
    if success:
        logger.info("All checkpoints downloaded successfully!")
        logger.info("\nCheckpoint files:")
        for name, info in checkpoints.items():
            size = info['dest'].stat().st_size / (1024 ** 2)  # Size in MB
            logger.info(f"  {info['dest']}: {size:.1f} MB")
    else:
        logger.error("Some downloads failed. Please check the errors above.")
    logger.info("="*60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
