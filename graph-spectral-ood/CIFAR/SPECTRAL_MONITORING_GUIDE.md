# Spectral Monitoring Integration Guide

## Overview
This guide explains how to use the spectral monitoring features that have been integrated into your Graph OOD detection training pipeline.

## What is Spectral Monitoring?

Spectral monitoring tracks the spectral gap of the k-NN graph constructed from feature representations. The spectral gap (λ₂ - λ₁) measures cluster structure:
- **High gap**: Clear cluster separation (good for OOD detection)
- **Low gap**: Poor cluster structure (degraded OOD detection)

## Key Features

### 1. **Spectral Gap Monitoring**
Tracks how the spectral gap changes when wild/auxiliary data is mixed with in-distribution data.

### 2. **Adaptive Mixture Adjustment**
Automatically adjusts π₁ (covariate shift) and π₂ (semantic shift) based on spectral degradation.

### 3. **Spectral Regularization**
Adds a loss term to maintain good spectral structure during training.

### 4. **Spectral Constraints**
Enforces minimum gap difference between in-distribution and out-of-distribution data.

## Usage

### Basic Monitoring
```bash
python train.py cifar10 \
    --spectral_monitor \
    --spectral_freq 10 \
    --spectral_k_neighbors 50
```

### With Adaptive Adjustment
```bash
python train.py cifar10 \
    --spectral_monitor \
    --spectral_adaptive \
    --start_epoch 50 \
    --pi_1 0.3 \
    --pi_2 0.2
```

### With Regularization
```bash
python train.py cifar10 \
    --spectral_monitor \
    --spectral_reg \
    --spectral_reg_alpha 0.01
```

### Full Configuration
```bash
python train.py cifar10 \
    --spectral_monitor \
    --spectral_freq 5 \
    --spectral_adaptive \
    --spectral_reg \
    --spectral_reg_alpha 0.01 \
    --spectral_k_neighbors 50 \
    --pi_1 0.3 \
    --pi_2 0.2 \
    --start_epoch 50
```

## Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--spectral_monitor` | flag | False | Enable spectral monitoring |
| `--spectral_freq` | int | 10 | Monitor every N epochs |
| `--spectral_adaptive` | flag | False | Enable adaptive π adjustment |
| `--spectral_reg` | flag | False | Add spectral regularization loss |
| `--spectral_reg_alpha` | float | 0.01 | Weight for spectral regularization |
| `--spectral_k_neighbors` | int | 50 | Number of neighbors for k-NN graph |
| `--spectral_analysis` | flag | False | Analyze spectral impact of different π combinations |

## Output Metrics

The spectral monitoring adds these metrics to your training state:

```python
state['spectral_gaps']        # List of spectral gap measurements
state['spectral_degradation'] # List of degradation percentages
```

### Console Output
During training, you'll see:
```
[Spectral Monitor] Epoch 10:
  Pure IN gap: 0.1234
  Mixed gap: 0.0987
  Degradation: 20.01%
  Wild ratio: 0.50
```

### Logged Metrics
- `gap_pure_in`: Spectral gap of pure in-distribution data
- `gap_mixed`: Spectral gap after mixing with wild data
- `degradation`: Relative gap reduction (0-1, lower is better)
- `wild_ratio`: Ratio of wild to in-distribution samples
- `effective_pi1`: Current π₁ value (covariate shift)
- `effective_pi2`: Current π₂ value (semantic shift)

## Integration with Existing Methods

### With SCONE/WOODS
```python
# The spectral constraint is automatically added when both are enabled
python train.py cifar10 \
    --score scone \
    --spectral_monitor \
    --spectral_reg
```

### With Energy Methods
```python
python train.py cifar10 \
    --score energy \
    --spectral_monitor \
    --m_in -25 \
    --m_out -5
```

## Spectral Analysis Feature

When `--spectral_analysis` is enabled, the system will periodically test different π combinations to find the optimal mixture that minimizes spectral degradation:

```
[Spectral Analysis] Epoch 50
π₁(cov)  π₂(label)  Gap(in)  Gap(mixed)  Degradation
-------------------------------------------------------
0.0      0.0        0.1234   0.1234      +0.0%
0.3      0.0        0.1234   0.1100      -10.9%
0.0      0.3        0.1234   0.0850      -31.1%
0.3      0.1        0.1234   0.1050      -14.9%
0.1      0.3        0.1234   0.0900      -27.1%

Best configuration: π₁=0.3, π₂=0.0
```

This analysis helps you understand:
- How covariate shift (π₁) affects clustering
- How semantic shift (π₂) affects clustering
- The optimal mixture for your specific dataset

## Advanced Usage

### Custom Monitoring Function
```python
from spectral_monitoring import compute_spectral_gap

# In your custom code
features = model.get_features(data)
gap = compute_spectral_gap(features, k=50)
print(f"Current spectral gap: {gap:.4f}")
```

### Spectral Regularization in Custom Loss
```python
from spectral_monitoring import spectral_regularization_loss

# Add to your loss computation
loss_spectral = spectral_regularization_loss(features, k=50, alpha=0.01)
total_loss = ce_loss + ood_loss + loss_spectral
```

### Adaptive Adjustment in Custom Training
```python
from spectral_monitoring import adaptive_mixture_adjustment

# After spectral monitoring
if spectral_stats['degradation'] > 0.3:
    args.pi_1, args.pi_2 = adaptive_mixture_adjustment(spectral_stats, args)
```

## Recommended Settings

### For CIFAR-10/100
```bash
--spectral_k_neighbors 50
--spectral_freq 10
--spectral_reg_alpha 0.01
```

### For MNIST
```bash
--spectral_k_neighbors 30
--spectral_freq 5
--spectral_reg_alpha 0.005
```

### For ImageNet
```bash
--spectral_k_neighbors 100
--spectral_freq 20
--spectral_reg_alpha 0.02
```

## Troubleshooting

### Issue: Spectral gap is always 0
- **Solution**: Increase `--spectral_k_neighbors` or check feature dimension

### Issue: High degradation throughout training
- **Solution**: Enable `--spectral_adaptive` to automatically adjust mixture

### Issue: Memory issues with large datasets
- **Solution**: Reduce sample_batches in spectral_monitoring_epoch()

### Issue: Slow computation
- **Solution**: Increase `--spectral_freq` to monitor less frequently

## Testing

Run the test script to verify installation:
```bash
python test_spectral.py
```

## References

The spectral monitoring is based on spectral graph theory for OOD detection. The spectral gap measures cluster quality which correlates with OOD detection performance.

## Support

For issues or questions, check the implementation in:
- `spectral_monitoring.py`: Core functions
- `train.py`: Integration points
- `test_spectral.py`: Test examples