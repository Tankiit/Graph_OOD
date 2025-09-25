#!/usr/bin/env python3
"""
Script to download SVHN, Textures (DTD), LSUN, LSUNR, and iSUN datasets
for the Graph Spectral OOD project.
"""

import os
import sys
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from pathlib import Path

# Add the project path to sys.path
project_root = Path(__file__).parent / "graph-spectral-ood"
sys.path.append(str(project_root))

# Import custom loaders
import utils.svhn_loader as svhn_loader
import utils.lsun_loader as lsun_loader

def download_svhn(data_path):
    """Download SVHN dataset"""
    print("Downloading SVHN dataset...")
    svhn_path = data_path / "svhn"
    svhn_path.mkdir(exist_ok=True)
    
    # Download SVHN files manually
    import urllib.request
    import scipy.io as sio
    
    # Download URLs and filenames
    urls = [
        ("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "train_32x32.mat"),
        ("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "test_32x32.mat"),
        ("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", "extra_32x32.mat")
    ]
    
    for url, filename in urls:
        filepath = svhn_path / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"{filename} already exists")
    
    # Load datasets to verify
    train_data = svhn_loader.SVHN(root=str(svhn_path), split="train")
    test_data = svhn_loader.SVHN(root=str(svhn_path), split="test")
    extra_data = svhn_loader.SVHN(root=str(svhn_path), split="extra")
    
    print(f"SVHN downloaded to: {svhn_path}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Extra samples: {len(extra_data)}")
    return svhn_path

def download_textures(data_path):
    """Download Describable Textures Dataset (DTD)"""
    print("Downloading DTD (Textures) dataset...")
    dtd_path = data_path / "dtd"
    dtd_path.mkdir(exist_ok=True)
    
    # Download DTD using torchvision
    dataset = dset.DTD(root=str(dtd_path), split="train", download=True, transform=None)
    
    print(f"DTD downloaded to: {dtd_path}")
    print(f"Total samples: {len(dataset)}")
    return dtd_path

def download_lsun(data_path):
    """Download LSUN dataset (LSUN_C)"""
    print("Downloading LSUN dataset...")
    lsun_c_path = data_path / "lsun_c"
    lsun_c_path.mkdir(exist_ok=True)
    
    print("Note: LSUN dataset requires manual download.")
    print("Please download LSUN bedroom images from: http://lsun.cs.princeton.edu/")
    print("Extract the bedroom_train_lmdb folder to:", lsun_c_path)
    print("The dataset should be organized as: lsun_c/bedroom_train_lmdb/")
    
    # Check if the dataset exists
    bedroom_path = lsun_c_path / "bedroom_train_lmdb"
    if bedroom_path.exists():
        try:
            dataset = lsun_loader.LSUN(db_path=str(lsun_c_path), classes=['bedroom_train'], transform=None)
            print(f"LSUN found at: {lsun_c_path}")
            print(f"Total samples: {len(dataset)}")
        except Exception as e:
            print(f"Error loading LSUN: {e}")
    else:
        print("LSUN dataset not found. Please download manually.")
    
    return lsun_c_path

def download_lsunr(data_path):
    """Download LSUN Resize dataset (LSUN_R)"""
    print("Downloading LSUN Resize dataset...")
    lsun_r_path = data_path / "lsun_r"
    lsun_r_path.mkdir(exist_ok=True)
    
    print("Note: LSUN Resize dataset requires manual download.")
    print("Please download LSUN bedroom validation images from: http://lsun.cs.princeton.edu/")
    print("Extract the bedroom_val_lmdb folder to:", lsun_r_path)
    print("The dataset should be organized as: lsun_r/bedroom_val_lmdb/")
    
    # Check if the dataset exists
    bedroom_path = lsun_r_path / "bedroom_val_lmdb"
    if bedroom_path.exists():
        try:
            dataset = lsun_loader.LSUN(db_path=str(lsun_r_path), classes=['bedroom_val'], transform=None)
            print(f"LSUN Resize found at: {lsun_r_path}")
            print(f"Total samples: {len(dataset)}")
        except Exception as e:
            print(f"Error loading LSUN Resize: {e}")
    else:
        print("LSUN Resize dataset not found. Please download manually.")
    
    return lsun_r_path

def download_isun(data_path):
    """Download iSUN dataset"""
    print("Downloading iSUN dataset...")
    isun_path = data_path / "isun"
    isun_path.mkdir(exist_ok=True)
    
    # Download iSUN - this is typically a subset of SUN dataset
    # We'll use ImageFolder to load it if it's already downloaded, or download SUN
    try:
        dataset = dset.ImageFolder(root=str(isun_path), transform=None)
        print(f"iSUN found at: {isun_path}")
        print(f"Total samples: {len(dataset)}")
    except:
        print("iSUN dataset not found. You may need to download it manually.")
        print("iSUN is typically a subset of the SUN dataset.")
        print("Please download iSUN from: http://isun.cs.princeton.edu/")
    
    return isun_path

def main():
    """Main function to download all datasets"""
    # Set up paths
    data_path = Path("graph-spectral-ood/data")
    data_path.mkdir(exist_ok=True)
    
    print("Starting dataset downloads...")
    print(f"Data will be saved to: {data_path.absolute()}")
    print("-" * 50)
    
    # Download datasets
    try:
        svhn_path = download_svhn(data_path)
        print("✓ SVHN download completed")
    except Exception as e:
        print(f"✗ SVHN download failed: {e}")
    
    try:
        textures_path = download_textures(data_path)
        print("✓ Textures (DTD) download completed")
    except Exception as e:
        print(f"✗ Textures download failed: {e}")
    
    try:
        lsun_path = download_lsun(data_path)
        print("✓ LSUN download completed")
    except Exception as e:
        print(f"✗ LSUN download failed: {e}")
    
    try:
        lsunr_path = download_lsunr(data_path)
        print("✓ LSUN Resize download completed")
    except Exception as e:
        print(f"✗ LSUN Resize download failed: {e}")
    
    try:
        isun_path = download_isun(data_path)
        print("✓ iSUN download completed")
    except Exception as e:
        print(f"✗ iSUN download failed: {e}")
    
    print("-" * 50)
    print("Dataset download process completed!")
    print(f"Check the data directory: {data_path.absolute()}")

if __name__ == "__main__":
    main()
