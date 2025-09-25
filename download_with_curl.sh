#!/bin/bash

# Dataset download script using curl
# This script downloads the remaining datasets for Graph Spectral OOD

DATA_DIR="/home/tanmoy/research/Graph_OOD/graph-spectral-ood/data"

echo "Starting dataset downloads with curl..."
echo "Data directory: $DATA_DIR"

# Function to download with curl and retry
download_with_curl() {
    local url=$1
    local output_file=$2
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "Attempting to download $output_file (attempt $((retry_count + 1)))"
        
        if curl -L --connect-timeout 30 --max-time 3600 -o "$output_file" "$url"; then
            echo "✓ Successfully downloaded $output_file"
            return 0
        else
            echo "✗ Download failed for $output_file"
            retry_count=$((retry_count + 1))
            sleep 5
        fi
    done
    
    echo "✗ Failed to download $output_file after $max_retries attempts"
    return 1
}

# Create directories
mkdir -p "$DATA_DIR/lsun_c"
mkdir -p "$DATA_DIR/lsun_r"
mkdir -p "$DATA_DIR/isun"

echo "=========================================="
echo "Downloading LSUN datasets..."

# LSUN bedroom training set (LSUN_C)
echo "Downloading LSUN bedroom training set..."
download_with_curl "https://www.dropbox.com/s/0hzg2jygb0lzhlf/bedroom_train_lmdb.zip?dl=1" "$DATA_DIR/lsun_c/bedroom_train_lmdb.zip"

# LSUN bedroom validation set (LSUN_R)  
echo "Downloading LSUN bedroom validation set..."
download_with_curl "https://www.dropbox.com/s/0hzg2jygb0lzhlf/bedroom_val_lmdb.zip?dl=1" "$DATA_DIR/lsun_r/bedroom_val_lmdb.zip"

echo "=========================================="
echo "Downloading iSUN dataset..."

# iSUN dataset
echo "Downloading iSUN dataset..."
download_with_curl "https://www.dropbox.com/s/agk2wj1y2jzxphj/iSUN.tar.gz?dl=1" "$DATA_DIR/isun/iSUN.tar.gz"

echo "=========================================="
echo "Extracting downloaded files..."

# Extract LSUN_C
if [ -f "$DATA_DIR/lsun_c/bedroom_train_lmdb.zip" ]; then
    echo "Extracting LSUN bedroom training set..."
    cd "$DATA_DIR/lsun_c"
    unzip -q bedroom_train_lmdb.zip
    rm bedroom_train_lmdb.zip
    echo "✓ LSUN_C extracted"
fi

# Extract LSUN_R
if [ -f "$DATA_DIR/lsun_r/bedroom_val_lmdb.zip" ]; then
    echo "Extracting LSUN bedroom validation set..."
    cd "$DATA_DIR/lsun_r"
    unzip -q bedroom_val_lmdb.zip
    rm bedroom_val_lmdb.zip
    echo "✓ LSUN_R extracted"
fi

# Extract iSUN
if [ -f "$DATA_DIR/isun/iSUN.tar.gz" ]; then
    echo "Extracting iSUN dataset..."
    cd "$DATA_DIR/isun"
    tar -xzf iSUN.tar.gz
    rm iSUN.tar.gz
    echo "✓ iSUN extracted"
fi

echo "=========================================="
echo "Dataset download and extraction completed!"
echo "Checking downloaded datasets..."

# Check what we have
ls -la "$DATA_DIR"

echo "=========================================="
echo "Summary:"
echo "- SVHN: ✓ Downloaded"
echo "- Textures (DTD): ✓ Downloaded" 
echo "- LSUN_C: Check if bedroom_train_lmdb folder exists"
echo "- LSUN_R: Check if bedroom_val_lmdb folder exists"
echo "- iSUN: Check if iSUN folder exists"
