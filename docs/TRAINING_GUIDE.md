# TimesNet Training Guide

## 📋 Overview

This guide shows how to train TimesNet on all ETT datasets with different prediction horizons.

**Training Setup:**
- **Datasets**: ETTh1, ETTh2, ETTm1, ETTm2
- **Input Length**: 96 (fixed, as in paper)
- **Prediction Horizons**: {24, 48, 96, 192, 336, 720}
- **Total Models**: 4 datasets × 6 horizons = 24 models

---

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook train.ipynb
```

Then run all cells sequentially. The notebook includes:
- ✅ Automatic training of all 24 models
- ✅ Results visualization (plots, heatmaps)
- ✅ Statistical analysis
- ✅ Comparison with paper results
- ✅ LaTeX table generation

### Option 2: Python Script
```bash
# Train all datasets and horizons
python train_all.py

# Train specific datasets
python train_all.py --datasets ETTh1 ETTh2

# Train specific horizons
python train_all.py --horizons 96 192 336

# Custom output directory
python train_all.py --output my_results
```

---

## 📊 What Gets Generated

### 1. Model Checkpoints
```
checkpoints/
├── ETTh1_96_24/checkpoint.pth
├── ETTh1_96_48/checkpoint.pth
├── ETTh1_96_96/checkpoint.pth
├── ETTh1_96_192/checkpoint.pth
├── ETTh1_96_336/checkpoint.pth
├── ETTh1_96_720/checkpoint.pth
├── ETTh2_96_24/checkpoint.pth
└── ... (24 total)
```

### 2. Results Files
```
results/
├── all_results.csv                      # Summary table
├── all_results_detailed.json            # Full results with training curves
├── error_vs_horizon.png                 # Error plots
├── mse_heatmap.png                      # MSE heatmap
├── mae_heatmap.png                      # MAE heatmap
├── results_table.tex                    # LaTeX table
└── {dataset}_{seq_len}_{pred_len}_curves.png  # Individual training curves (24 files)
```

---

## 🎯 Model Configurations

### Paper Hyperparameters

#### ETTh (Hourly Data)
```python
ETTh1/ETTh2:
    d_model = 16      # Small model
    d_ff = 32
    top_k = 5
    e_layers = 2
    batch_size = 32
    learning_rate = 0.0001
    epochs = 10
```

#### ETTm (15-minute Data)
```python
ETTm1/ETTm2:
    d_model = 32      # Larger model
    d_ff = 64
    top_k = 5
    e_layers = 2
    batch_size = 32
    learning_rate = 0.0001
    epochs = 10
```

### All Datasets Share
```python
seq_len = 96          # Input: 96 timesteps
enc_in = 7            # 7 input features
c_out = 7             # 7 output features
top_k = 5             # Top-5 frequencies
num_kernels = 6       # Inception kernels
dropout = 0.1
embed = 'fixed'
patience = 3          # Early stopping
```

---

## 📈 Expected Results

### ETTh1 (Reference from Paper)
| Horizon | MSE | MAE |
|---------|-----|-----|
| 96 | 0.384 | 0.402 |
| 192 | 0.436 | 0.429 |
| 336 | 0.491 | 0.469 |
| 720 | 0.521 | 0.491 |

### ETTm1 (Reference from Paper)
| Horizon | MSE | MAE |
|---------|-----|-----|
| 96 | 0.334 | 0.365 |
| 192 | 0.374 | 0.385 |
| 336 | 0.410 | 0.403 |
| 720 | 0.478 | 0.437 |

*Note: Your results may vary slightly due to random initialization*

---

## 🔧 Customization

### Train Single Model
```python
from src.train import TimesNetTrainer

config = {
    'root_path': './ETDataset/ETT-small/',
    'data_path': 'ETTh1.csv',
    'seq_len': 96,
    'pred_len': 96,
    'enc_in': 7,
    'c_out': 7,
    'd_model': 16,
    'd_ff': 32,
    'top_k': 5,
    'e_layers': 2,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'train_epochs': 10,
}

trainer = TimesNetTrainer(config)
trainer.train()
results = trainer.test()
```

### Different Horizons
```python
# Short-term (1 day)
pred_len = 24

# Medium-term (4 days)
pred_len = 96

# Long-term (30 days)
pred_len = 720
```

### Different Datasets
```python
# Hourly data
data_path = 'ETTh1.csv'  # or 'ETTh2.csv'
d_model = 16
d_ff = 32

# 15-minute data
data_path = 'ETTm1.csv'  # or 'ETTm2.csv'
d_model = 32
d_ff = 64
```

---

## 📊 Results Analysis (Notebook Only)

The Jupyter notebook provides comprehensive analysis:

### 1. **Individual Dataset Results**
- Performance for each dataset across all horizons
- Training curves for each model

### 2. **Comparative Visualizations**
- Error vs Prediction Horizon plots
- MSE/MAE heatmaps
- Cross-dataset comparisons

### 3. **Statistical Analysis**
- Summary statistics
- Best models per dataset
- Comparison with paper results

### 4. **Export Formats**
- CSV for Excel/spreadsheet analysis
- JSON for programmatic access
- LaTeX table for papers/reports
- PNG plots for presentations

---

## ⏱️ Training Time

Approximate training time per model (on GPU):

| Dataset | Horizon | Time |
|---------|---------|------|
| ETTh | 24-96 | ~5 min |
| ETTh | 192-336 | ~7 min |
| ETTh | 720 | ~10 min |
| ETTm | 24-96 | ~8 min |
| ETTm | 192-336 | ~12 min |
| ETTm | 720 | ~15 min |

**Total time for all 24 models: ~3-4 hours on GPU, ~10-12 hours on CPU**

---

## 🐛 Troubleshooting

### GPU Out of Memory
```python
# Reduce batch size
config['batch_size'] = 16  # or 8

# Or use CPU
config['device'] = 'cpu'
```

### Training Too Slow
```python
# Use mixed precision (if GPU available)
config['use_amp'] = True

# Reduce epochs
config['train_epochs'] = 5

# Increase patience for faster early stopping
config['patience'] = 2
```

### Poor Results
```python
# Increase epochs
config['train_epochs'] = 20

# Adjust learning rate
config['learning_rate'] = 0.0005  # or 0.00005

# Check data normalization
# Ensure scale=True in dataset
```

---

## 📝 Results Format

### CSV (all_results.csv)
```csv
dataset,seq_len,pred_len,d_model,d_ff,test_mse,test_mae,test_rmse,final_epoch
ETTh1,96,24,16,32,0.3821,0.4015,0.6181,8
ETTh1,96,48,16,32,0.4102,0.4234,0.6405,9
...
```

### JSON (all_results_detailed.json)
```json
[
  {
    "dataset": "ETTh1",
    "seq_len": 96,
    "pred_len": 24,
    "d_model": 16,
    "d_ff": 32,
    "train_losses": [0.512, 0.423, 0.398, ...],
    "val_losses": [0.445, 0.401, 0.389, ...],
    "test_mse": 0.3821,
    "test_mae": 0.4015,
    "test_rmse": 0.6181,
    "final_epoch": 8,
    "timestamp": "2024-01-15T10:30:00"
  },
  ...
]
```

---

## 📚 What Each Horizon Means

| Horizon | ETTh (hourly) | ETTm (15-min) | Use Case |
|---------|---------------|---------------|----------|
| 24 | 1 day | 6 hours | Very short-term |
| 48 | 2 days | 12 hours | Short-term |
| 96 | 4 days | 1 day | Medium-term |
| 192 | 8 days | 2 days | Long-term |
| 336 | 14 days | 3.5 days | Very long-term |
| 720 | 30 days | 7.5 days | Ultra long-term |

---

## ✅ Validation

After training, verify results:

1. **Check MSE/MAE values** are reasonable (0.3-0.6 range for ETT)
2. **Compare with paper** (should be within ±10%)
3. **Look at training curves** (should converge smoothly)
4. **Verify early stopping** (should stop before max epochs)

---

## 🎉 Next Steps

After training:

1. **Analyze results** in the notebook
2. **Compare with paper** to validate implementation
3. **Generate plots** for your report/presentation
4. **Export LaTeX table** for papers
5. **Use best models** for predictions

---

## 📖 Related Documentation

- [README_TIMESNET.md](README_TIMESNET.md) - Main implementation guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick start guide
- [PARAMETERS_GUIDE.md](PARAMETERS_GUIDE.md) - Parameter reference
- [ARCHITECTURE_EXPLAINED.md](ARCHITECTURE_EXPLAINED.md) - Architecture details

---

**Happy Training! 🚀**
