#!/bin/bash

# Script to setup Jupyter notebook with torch-multimodal conda environment

echo "Setting up Jupyter notebook with torch-multimodal environment..."

# Activate the conda environment
conda activate torch-multimodal

# Install ipykernel in the environment if not already installed
conda install -y ipykernel

# Add the environment as a Jupyter kernel
python -m ipykernel install --user --name torch-multimodal --display-name "Python (torch-multimodal)"

echo "✅ Setup complete!"
echo ""
echo "To use the environment in Jupyter:"
echo "1. Start Jupyter: jupyter notebook"
echo "2. Open timm_knn_features.ipynb"
echo "3. In the notebook, go to Kernel > Change kernel > Python (torch-multimodal)"
echo ""
echo "Or run Jupyter directly with the environment:"
echo "conda activate torch-multimodal && jupyter notebook timm_knn_features.ipynb"