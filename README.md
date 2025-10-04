# TimesNet for ETT Forecasting

PyTorch implementation of TimesNet for long-term time series forecasting on ETT datasets.

## What is TimesNet?

TimesNet converts 1D time series into 2D representations based on discovered periodicities, then uses 2D convolutions to capture temporal patterns. It's simpler than Transformers - no attention, no autoregressive decoding.

**Key idea:** Find dominant periods (daily, weekly) via FFT, reshape the sequence into 2D grids for each period, apply multi-scale 2D convolutions.

### Architecture

```
Input [96 timesteps]
  → Normalize (zero mean, unit variance)
  → Embed (value + time + position)
  → Linear expand (96 → 192 timesteps)
  → TimesBlocks ×2:
      - FFT finds top-5 periods
      - Reshape to 2D for each period
      - Multi-scale 2D conv (Inception-style)
      - Adaptive weighted aggregation
  → Project back to features
  → Extract last 96 timesteps as predictions
```

No decoder. No future time features. Just encoder with temporal expansion.

See [docs/ARCHITECTURE_EXPLAINED.md](docs/ARCHITECTURE_EXPLAINED.md) for detailed flow.

## Repository Structure

```
timesnet-ett/
├── src/
│   ├── dataset.py       # ETT data loading (70/10/20 splits)
│   ├── model.py         # TimesNet architecture
│   ├── train.py         # Training loop with early stopping
│   └── evaluate.py      # Metrics + visualizations
├── train.ipynb          # Main training notebook (all datasets)
├── train_v2.ipynb       # Hyperparameter experiments
├── test_train.py        # Quick validation test
└── docs/
    └── ARCHITECTURE_EXPLAINED.md
```

## Quick Start

```bash
# Install dependencies
pip install torch pandas scikit-learn matplotlib seaborn

# Validate implementation
python test_train.py

# Train on ETTh1
python src/train.py
```

## Training Notebooks

### `train.ipynb` - Full Training
Trains 24 models (4 datasets × 6 horizons) with comprehensive visualizations:
- Horizon predictions (input context + forecast)
- 4-panel training summary (loss curves, gap analysis, metrics)
- Train/val/test evaluation

**Outputs per model:**
- `results/{dataset}_{horizon}_horizons.png` - Visual predictions
- `results/{dataset}_{horizon}_summary.png` - Training analysis
- `results/{dataset}_{horizon}_all_metrics.txt` - Full metrics

### `train_v2.ipynb` - Hyperparameter Experiments
Tests different configurations to improve performance:

| Experiment | Changes | Hypothesis |
|------------|---------|------------|
| V1 (Baseline) | Paper config: LR decay every epoch | Current setup |
| V2 (Constant LR) | Fixed LR=0.0001, 30 epochs, patience=5 | LR decay too aggressive |
| V3 (Larger Model) | d_model=32, d_ff=64 | Model undercapacity |
| V4 (Best Combo) | Larger model + LR=0.00005 + 50 epochs | Combined improvements |

**Why these experiments?**

The baseline uses `lradj='type1'` which halves LR every epoch:
```
Epoch 1: LR = 0.0001
Epoch 2: LR = 0.00005  (already 50% smaller!)
Epoch 3: LR = 0.000025 (too small to learn effectively)
```

This might explain why validation loss stagnates - the model doesn't have time to converge before LR becomes tiny.

## Implementation Notes

### What's Different from Time-Series-Library?

1. **Encoder-only design** - No decoder inputs (`x_dec`, `x_mark_dec`, `dec_inp`)
2. **3-item dataset** - Returns `(seq_x, seq_y, seq_x_mark)` not 4 items
3. **No future time features** - Model only sees input time features, learns to extrapolate
4. **Simplified interface** - `model(x_enc, x_mark_enc)` with 2 arguments

This matches the paper's TimesNet for forecasting (not the generic wrapper).

### Dataset Configuration

- **Splits:** 70% train, 10% val, 20% test (could use fixed months like paper)
- **Normalization:** StandardScaler fit on train data
- **Time encoding:** Manual (month, day, weekday, hour) with fixed embeddings
- **No overlap:** Input and target are consecutive, non-overlapping sequences

### Model Hyperparameters (ETTh1)

From paper:
```python
seq_len = 96        # Input: 4 days hourly
pred_len = 96       # Output: next 4 days
d_model = 16        # Small model for ETT
d_ff = 32
top_k = 5           # Top-5 periods
e_layers = 2        # 2 TimesBlocks
num_kernels = 6     # Multi-scale inception
dropout = 0.1
```

**Total parameters:** ~600K

## Visualizations

### Horizon Predictions
Shows input context (gray) + predictions (red) vs ground truth (green) for train/val/test:
```python
visualize_horizon_predictions(
    model, train_loader, val_loader, test_loader,
    device, scaler, seq_len, pred_len
)
```

### Training Summary
4-panel plot:
1. Loss curves (train/val/test)
2. Generalization gap (val - train)
3. Test metrics bar chart
4. Summary statistics

### What to Look For

**Good model:**
- Val loss decreases with train loss
- Small generalization gap (<0.05)
- Predictions follow trends, not just mean
- Similar performance across splits

**Problematic:**
- Val loss stagnates while train decreases → overfitting or LR issue
- Large gap (>0.1) → overfitting
- Flat predictions → model not learning patterns

## Known Issues & Solutions

### Issue 1: Val Loss Stagnation
**Symptom:** Training loss decreases, validation loss stays constant

**Likely cause:** Learning rate schedule too aggressive (`lradj='type1'`)

**Solution:** Use `train_v2.ipynb` to test constant LR

### Issue 2: Results Don't Match Paper
**Check:**
1. Data splits (70/10/20 vs fixed months?)
2. Learning rate schedule (constant vs decay?)
3. Training duration (10 epochs might be insufficient)
4. Model capacity (d_model=16 might be too small)

Run experiments in `train_v2.ipynb` to identify the issue.

## Results Format

CSV output includes:
```
dataset, pred_len, d_model, d_ff,
train_mse, train_mae,
val_mse, val_mae,
test_mse, test_mae, test_rmse,
final_epoch
```

Compare with paper Table 1:
- ETTh1 96→96: MSE=0.384, MAE=0.402
- ETTh1 96→192: MSE=0.436, MAE=0.429
- etc.

## Validation

Run `test_train.py` to verify implementation:
```bash
python test_train.py
```

Checks:
- Dataset returns 3 items (not 4)
- Model is encoder-only (rejects decoder inputs)
- Forward pass: 96 → 192 → extract last 96
- All steps match architecture documentation

## References

- Paper: [TimesNet (ICLR 2023)](https://openreview.net/pdf?id=ju_Uqw384Oq)
- Code: [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- Dataset: [ETT (Electricity Transformer Temperature)](https://github.com/zhouhaoyi/ETDataset)

## File Summary

**Core:**
- `src/model.py` - TimesNet implementation
- `src/dataset.py` - ETT data loader
- `src/train.py` - Training with early stopping
- `src/evaluate.py` - Metrics + visualization functions

**Training:**
- `train.ipynb` - Production training (24 models)
- `train_v2.ipynb` - Hyperparameter experiments

**Docs:**
- `docs/ARCHITECTURE_EXPLAINED.md` - Detailed architecture walkthrough
- `BUG_FIX_SUMMARY.md` - Implementation analysis & debugging guide
- `NOTEBOOK_UPDATES.md` - What changed in notebooks

**Testing:**
- `test_train.py` - Quick validation

---

Built for UTEC Deep Learning course. Implementation validated against paper architecture and Time-Series-Library reference.
