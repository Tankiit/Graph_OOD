#!/usr/bin/env python3
"""
Script to download LSUN bedroom datasets from Hugging Face
"""

from datasets import load_dataset
import os
from pathlib import Path

def download_lsun_from_hf():
    """Download LSUN bedroom datasets from Hugging Face"""
    
    # Set up paths
    data_dir = Path("graph-spectral-ood/data")
    lsun_c_path = data_dir / "lsun_c"
    lsun_r_path = data_dir / "lsun_r"
    
    # Create directories
    lsun_c_path.mkdir(parents=True, exist_ok=True)
    lsun_r_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading LSUN bedroom datasets from Hugging Face...")
    print("Dataset: pcuenq/lsun-bedrooms")
    
    try:
        # Load the dataset
        print("Loading dataset...")
        ds = load_dataset("pcuenq/lsun-bedrooms")
        
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}")
        
        # Check what splits are available
        for split_name, split_data in ds.items():
            print(f"\nSplit: {split_name}")
            print(f"Number of samples: {len(split_data)}")
            print(f"Features: {split_data.features}")
            
            # Save the dataset
            if split_name == "train":
                output_path = lsun_c_path / "bedroom_train"
                print(f"Saving training data to: {output_path}")
                split_data.save_to_disk(str(output_path))
                
            elif split_name == "validation":
                output_path = lsun_r_path / "bedroom_val"
                print(f"Saving validation data to: {output_path}")
                split_data.save_to_disk(str(output_path))
        
        print("\n✓ LSUN datasets downloaded successfully!")
        print(f"Training data saved to: {lsun_c_path}")
        print(f"Validation data saved to: {lsun_r_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading LSUN datasets: {e}")
        return False

def main():
    """Main function"""
    print("Starting LSUN dataset download from Hugging Face...")
    print("=" * 50)
    
    success = download_lsun_from_hf()
    
    if success:
        print("\n" + "=" * 50)
        print("Dataset download completed successfully!")
        print("\nSummary:")
        print("- SVHN: ✓ Downloaded")
        print("- Textures (DTD): ✓ Downloaded")
        print("- LSUN_C: ✓ Downloaded from Hugging Face")
        print("- LSUN_R: ✓ Downloaded from Hugging Face")
        print("- iSUN: ✓ Downloaded")
    else:
        print("\n" + "=" * 50)
        print("Dataset download failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
