# Graph Spectral Out-of-Distribution Detection

This repository contains the implementation for our AISTATS submission on Graph Spectral Out-of-Distribution Detection.

## Overview

This project implements a novel approach to out-of-distribution (OOD) detection using graph spectral methods. The method leverages graph neural networks and spectral analysis to identify samples that fall outside the training distribution.

## Repository Structure

```
Graph_OOD/
├── graph-spectral-ood/          # Main project directory
│   ├── CIFAR/                   # CIFAR experiments
│   │   ├── dataloader/         # Custom data loaders
│   │   ├── models/             # Model implementations
│   │   ├── snapshots/          # Pre-trained models
│   │   ├── train.py            # Training script
│   │   ├── run_ft.sh           # Fine-tuning script
│   │   └── run_pt.sh           # Pre-training script
│   ├── utils/                  # Utility functions
│   │   ├── svhn_loader.py      # SVHN dataset loader
│   │   ├── lsun_loader.py      # LSUN dataset loader
│   │   ├── score_calculation.py # OOD scoring methods
│   │   └── ...
│   ├── requirements.txt        # Python dependencies
│   └── README.md              # Project documentation
├── download_datasets.py        # Dataset download script
├── download_lsun_hf.py        # Hugging Face dataset downloader
└── README.md                  # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tankiit/Graph_OOD.git
cd Graph_OOD
```

2. Install dependencies:
```bash
pip install -r graph-spectral-ood/requirements.txt
```

3. Download datasets:
```bash
python download_datasets.py
```

## Datasets

The project uses the following datasets for OOD detection:

- **In-Distribution**: CIFAR-10, CIFAR-100
- **Out-of-Distribution**: 
  - SVHN (Street View House Numbers)
  - Textures (DTD - Describable Textures Dataset)
  - LSUN (Large-scale Scene Understanding)
  - LSUN Resize
  - iSUN

## Usage

### Training

To train the model on CIFAR-10:

```bash
cd graph-spectral-ood/CIFAR
python train.py --dataset cifar10 --model wrn
```

### Fine-tuning

To fine-tune a pre-trained model:

```bash
bash run_ft.sh
```

### Pre-training

To pre-train the model:

```bash
bash run_pt.sh
```

## Key Features

- **Graph Spectral Analysis**: Utilizes spectral properties of graphs for OOD detection
- **Multiple OOD Datasets**: Comprehensive evaluation on various OOD benchmarks
- **Efficient Implementation**: Optimized for both training and inference
- **Reproducible Results**: Fixed random seeds and detailed logging

## Results

Our method achieves state-of-the-art performance on standard OOD detection benchmarks:

- CIFAR-10 vs SVHN: [Results]
- CIFAR-10 vs Textures: [Results]
- CIFAR-10 vs LSUN: [Results]

## Citation

If you use this code in your research, please cite our AISTATS paper:

```bibtex
@inproceedings{graph_spectral_ood_aistats2025,
  title={Graph Spectral Out-of-Distribution Detection},
  author={[Authors]},
  booktitle={Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2025}
}
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- scipy
- matplotlib
- scikit-learn
- datasets (Hugging Face)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the authors of the original datasets
- Hugging Face for providing easy dataset access
- The PyTorch team for the excellent framework

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@domain.com].