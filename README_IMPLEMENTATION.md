# PaCo-2: Part-aware Contrast with Second-order Consistency

Implementation of PaCo-2 for Ultra-Fine-Grained Visual Categorization (UFGVC) based on the research design in `docs/`.

## Overview

PaCo-2 combines:
- **Part-aware Contrast (PaC)**: Triplet-style contrastive loss on matched parts with Mahalanobis distance
- **Second-order Consistency (SoC)**: Minimizes covariance difference between two views with optional prototype regularization
- **Timm Integration**: Seamless support for all timm CNN backbones

## Project Structure

```
PaCo-2/
├── src/
│   ├── dataset/
│   │   └── ufgvc.py          # UFGVC dataset implementation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── paco_model.py     # Main PaCo-2 model
│   │   ├── losses.py         # PaC, SoC, and combined losses
│   │   └── utils.py          # Part sampling, Hungarian matching, covariance utils
│   ├── train_utils.py        # Training utilities and metrics
│   └── data_utils.py         # Data augmentation and transforms
├── configs/
│   ├── ufg_base.yaml         # Base configuration
│   ├── ufg_convnext.yaml     # ConvNeXt configuration
│   ├── ufg_efficientnet.yaml # EfficientNet configuration
│   └── ablation_soc_metric.yaml # Ablation study config
├── scripts/
│   ├── train_a100.sh         # A100 training script
│   ├── train_v100.sh         # V100 training script
│   └── test.sh               # Testing script
├── docs/                     # Design documents
├── train_clean.py            # Main training script
├── evaluate.py               # Evaluation script
└── requirements.txt
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The UFGVC datasets will be automatically downloaded when first used.

## Usage

### Training

#### Basic Training
```bash
python train_clean.py --config configs/ufg_base.yaml
```

#### Different Backbones
```bash
# ConvNeXt-T
python train_clean.py --config configs/ufg_convnext.yaml

# EfficientNet-B3
python train_clean.py --config configs/ufg_efficientnet.yaml
```

#### Resume Training
```bash
python train_clean.py --config configs/ufg_base.yaml --resume runs/cotton80/resnet50/.../best_model.pth
```

#### Custom Device
```bash
python train_clean.py --config configs/ufg_base.yaml --device cuda:1
```

### Evaluation

```bash
python evaluate.py --checkpoint runs/cotton80/resnet50/.../best_model.pth --config configs/ufg_base.yaml --dataset test
```

### Using Job Scripts (for Gadi)

```bash
# Submit A100 job
cd scripts
qsub train_a100.sh

# Submit V100 job  
qsub train_v100.sh
```

## Configuration

### Core Parameters (from `docs/exp_data.md`)

```yaml
model:
  # Part sampling
  K: 4              # Number of parts
  r: 5              # Part window size
  d: 64             # Reduced dimension
  
  # Loss weights
  lambda_pac: 1.0   # PaC loss weight
  eta_soc: 0.1      # SoC loss weight
  alpha: 0.2        # Triplet margin
  beta: 0.05        # Prototype regularization
  
  # Technical parameters
  epsilon: 1.0e-5   # Covariance diagonal loading
  tau: 1.0e-5       # Sigma_plus regularization
  metric: "fro"     # SPD distance: "fro", "log-euclidean", "stein"
```

### Available Datasets

- `cotton80`: Cotton classification (80 classes)
- `soybean`: Soybean classification  
- `soy_ageing_r1` to `soy_ageing_r6`: Soybean aging datasets

### Supported Backbones (timm)

- `resnet50`, `resnet101`
- `convnext_tiny`, `convnext_small`, `convnext_base`
- `efficientnet_b0` to `efficientnet_b7`
- `mobilenetv3_large_100`
- `regnety_008`, `regnety_016`
- Any other timm model

## Key Features

### 1. Part-aware Contrast (PaC)
- Extracts K salient parts using top-K peak detection
- Hungarian matching for part correspondence between views
- Mahalanobis distance-based triplet loss
- Semi-hard negative mining

### 2. Second-order Consistency (SoC)
- Low-rank covariance estimation with diagonal loading
- SPD manifold distances (Frobenius, Log-Euclidean, Stein)
- Optional class prototype regularization

### 3. Technical Optimizations
- Cholesky decomposition for numerical stability
- Cosine to Mahalanobis warmup scheduling
- Memory-efficient batch-wise negative sampling
- EMA-based class prototype updates

### 4. Data Augmentation
- Light occlusion and patch shuffling (following CLE-ViT)
- RandAugment with controlled magnitude for fine-grained tasks
- Two-view transforms for contrastive learning

## Experimental Setup

Following `docs/exp_data.md`:

### Core Comparison
- **Baseline**: CE only
- **+Contrast**: CE + SupCon  
- **CLE-style**: CE + Triplet
- **Ours**: CE + PaC + SoC

### Ablation Studies
- SoC distance metrics: Frobenius vs Log-Euclidean vs Stein
- PaC construction: Cosine vs Mahalanobis matching
- Hyperparameter sweeps: K, d, r, λ, η, β
- Class prototype usage
- Warmup strategies

## Results Directory Structure

```
runs/
└── {dataset}/
    └── {model}/
        └── {timestamp}/
            ├── config.yaml           # Training configuration
            ├── metrics.csv           # Training metrics
            ├── curves/               # Training curves
            ├── checkpoints/          # Model checkpoints
            ├── best_model.pth        # Best model
            └── final_info.json       # Final results
```

## Model Architecture

```python
from src.models import PaCoModel

# Create model
model = PaCoModel(
    backbone_name='resnet50',
    num_classes=80,
    K=4, r=5, d=64,
    lambda_pac=1.0, eta_soc=0.1
)

# Forward pass (training)
outputs = model(view1, view2, targets)
# outputs: {'logits': ..., 'total': ..., 'ce': ..., 'pac': ..., 'soc': ...}

# Forward pass (inference)  
outputs = model(image)
# outputs: {'logits': ...}
```

## Implementation Details

### Following timm Requirements
- Uses `timm.create_model()` for backbone creation
- Calls `model.forward_features()` for feature extraction
- Follows timm data transforms and configurations

### Memory Efficiency
- K and d are small constants (K=4, d=64)
- Batch-wise negative sampling (no memory bank)
- Low-rank covariance computation O(K·d²)

### Numerical Stability
- Diagonal loading: Σ ← Σ + εI
- Cholesky decomposition for Mahalanobis distance
- Regularized Sigma_plus: Σ_plus ← 0.5(Σ₁+Σ₂) + τI

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch_size or use smaller backbone
2. **Cholesky decomposition failed**: Increase epsilon value
3. **No improvement**: Check warmup_epochs and loss weights
4. **Dataset download issues**: Check network connection and disk space

### Debug Mode
Set smaller epochs and print_freq for debugging:
```yaml
training:
  epochs: 5
  print_freq: 10
```

## Citation

Based on the PaCo-2 research design documented in `docs/abs.md`.

## License

This implementation follows the experimental design and pseudo code specifications provided in the documentation.
