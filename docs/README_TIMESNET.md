# TimesNet Implementation - Complete Guide

## 📂 Project Structure

```
project_2/
├── src/
│   ├── dataset.py          # ETT dataset loading (simplified)
│   ├── model.py            # TimesNet model architecture
│   ├── train.py            # Training pipeline
│   └── evaluate.py         # Evaluation metrics
│
├── ETDataset/
│   └── ETT-small/
│       ├── ETTh1.csv       # Hourly data
│       ├── ETTh2.csv
│       ├── ETTm1.csv       # 15-minute data
│       └── ETTm2.csv
│
├── Documentation/
│   ├── README_TIMESNET.md  ← You are here
│   ├── QUICK_REFERENCE.md  # Quick start guide
│   ├── PARAMETERS_GUIDE.md # Complete parameter reference
│   ├── ARCHITECTURE_EXPLAINED.md  # Visual architecture guide
│   ├── SIMPLIFIED_CHANGES.md      # What changed and why
│   └── UPDATE_SUMMARY.md          # Update summary
│
└── test_implementation.py  # Validation script
```

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

### Step 2: Run Training
```python
from src.train import train_timesnet

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
}

trainer, results = train_timesnet(config)
print(f"Test MSE: {results['mse']:.4f}")
print(f"Test MAE: {results['mae']:.4f}")
```

### Step 3: Validate
```bash
python test_implementation.py
```

---

## 📖 Documentation Guide

### New to TimesNet? Start here:
1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Copy-paste examples and basics
2. **[ARCHITECTURE_EXPLAINED.md](ARCHITECTURE_EXPLAINED.md)** - How TimesNet works visually

### Need parameter details?
3. **[PARAMETERS_GUIDE.md](PARAMETERS_GUIDE.md)** - Complete parameter reference
   - What each parameter does
   - Paper values
   - Common mistakes

### Understanding the changes?
4. **[SIMPLIFIED_CHANGES.md](SIMPLIFIED_CHANGES.md)** - What was removed and why
5. **[UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)** - Complete update summary

---

## ✨ What's Different (Simplified Implementation)

### ✅ What We Kept:
- Full TimesNet architecture (FFT + 2D Conv)
- Paper hyperparameters (d_model=16, top_k=5, etc.)
- Normalization and time encoding
- Training pipeline with early stopping

### ❌ What We Removed:
- `label_len` - Decoder overlap (TimesNet has no decoder!)
- `features` - Forecasting mode (always multivariate)
- `target` - Target column selection (not needed)
- Decoder inputs (x_dec, x_mark_dec)

### 🎯 Result:
**Simpler, cleaner, paper-faithful implementation!**

---

## 📊 Key Concepts Explained

### 1. **`scale` Parameter** (Data Normalization)
```python
if scale=True:
    # Normalize to zero mean, unit variance
    scaler.fit(train_data)           # Fit on train only
    data = scaler.transform(all_data)  # Transform all
```
**Why**: Different features have different scales → helps training
**Paper uses**: `True` ✅

### 2. **`timeenc` Parameter** (Time Feature Encoding)

#### Manual (0) - Default:
```python
# Categorical time features → embeddings
[month, day, weekday, hour]  # integers
```

#### Fourier (1) - Alternative:
```python
# Continuous time features → linear layer
[hour/23 - 0.5, dayofweek/6 - 0.5, ...]  # floats
```

**Purpose**: Tell model "what time it is"
**NOT the same as**: FFT period detection (that's in the model!)
**Paper uses**: `0` (manual) ✅

### 3. **FFT Period Detection** (Model Core)
```python
# In model.py - TimesNet's innovation!
def FFT_for_Period(x, k=5):
    xf = torch.fft.rfft(x, dim=1)
    amplitudes = abs(xf).mean(0).mean(-1)
    top_k_freqs = topk(amplitudes, k)
    periods = seq_len // top_k_freqs
    return periods, amplitudes
```

**Purpose**: Find dominant periods (daily, weekly, etc.)
**Used for**: 2D reshaping and multi-period modeling
**This is what makes TimesNet work!**

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  TIMESNET                        │
│                                                  │
│  Input [B,96,7]                                  │
│    ↓                                             │
│  Normalization (subtract mean, divide std)       │
│    ↓                                             │
│  Embedding (value + time + position) [B,96,16]   │
│    ↓                                             │
│  Linear Projection [B,96,16] → [B,144,16]        │
│    ↓                                             │
│  TimesBlock ×2:                                  │
│    1. FFT → Find top-5 periods                   │
│    2. Reshape to 2D based on periods             │
│    3. Multi-scale 2D Conv (Inception)            │
│    4. Adaptive aggregation by amplitude          │
│    5. Residual connection                        │
│    ↓                                             │
│  Layer Norm [B,144,16]                           │
│    ↓                                             │
│  Projection [B,144,16] → [B,144,7]               │
│    ↓                                             │
│  De-normalization (×std + mean)                  │
│    ↓                                             │
│  Extract predictions [B,144,7][-96:] → [B,96,7]  │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Key**: Encoder-only, no decoder needed!

---

## 📝 Usage Examples

### Example 1: Different Prediction Horizons
```python
# 96 → 192 (8 days)
config = {
    'root_path': './ETDataset/ETT-small/',
    'data_path': 'ETTh1.csv',
    'seq_len': 96,
    'pred_len': 192,  # Change this!
    'enc_in': 7,
    'c_out': 7,
    'd_model': 16,
    'd_ff': 32,
    'top_k': 5,
    'e_layers': 2,
}
```

### Example 2: Different Dataset
```python
# ETTm1 (15-minute data)
config = {
    'root_path': './ETDataset/ETT-small/',
    'data_path': 'ETTm1.csv',  # Change to ETTm1
    'seq_len': 96,
    'pred_len': 96,
    'enc_in': 7,
    'c_out': 7,
    'd_model': 32,  # Larger for ETTm
    'd_ff': 64,     # Larger for ETTm
    'top_k': 5,
    'e_layers': 2,
}
```

### Example 3: Manual Training Loop
```python
from src.dataset import create_dataloaders
from src.model import TimesNet

# Load data
train_loader, val_loader, test_loader, dataset = create_dataloaders(
    root_path='./ETDataset/ETT-small/',
    data_path='ETTh1.csv',
    seq_len=96,
    pred_len=96
)

# Create model
model = TimesNet(
    seq_len=96, pred_len=96,
    enc_in=7, c_out=7,
    d_model=16, d_ff=32,
    top_k=5, e_layers=2
)

# Training loop
for batch_x, batch_y, batch_x_mark in train_loader:  # 3 items!
    outputs = model(batch_x, batch_x_mark)  # 2 args!
    # ... training code
```

---

## 🔬 Paper Results (Reference)

### ETTh1 Results
| Horizon | MSE | MAE |
|---------|-----|-----|
| 96 | 0.384 | 0.402 |
| 192 | 0.436 | 0.429 |
| 336 | 0.491 | 0.469 |
| 720 | 0.521 | 0.491 |

### ETTm1 Results
| Horizon | MSE | MAE |
|---------|-----|-----|
| 96 | 0.334 | 0.365 |
| 192 | 0.374 | 0.385 |
| 336 | 0.410 | 0.403 |
| 720 | 0.478 | 0.437 |

*Source: TimesNet paper (Table 1)*

---

## ❓ FAQ

### Q: Why no `label_len` parameter?
**A**: TimesNet doesn't use a decoder. It directly projects from input to output length, so no decoder initialization (label_len) is needed.

### Q: What's the difference between `timeenc` and FFT in the model?
**A**:
- `timeenc`: Encodes calendar info (hour, day, etc.) as input features
- Model FFT: Finds periodic patterns in the data for 2D modeling
- Completely different purposes!

### Q: Should I use `timeenc=0` or `timeenc=1`?
**A**: Use `timeenc=0` (manual encoding) - it's what the paper uses and works well.

### Q: Why such small `d_model` for ETT?
**A**: ETT is a simple dataset. Small models (d_model=16) prevent overfitting and train faster. Use larger values (32, 64) for complex datasets.

### Q: Can I change `top_k`?
**A**: Yes, but `top_k=5` works well for most datasets. It's a key hyperparameter - too small misses patterns, too large adds noise.

---

## 🐛 Troubleshooting

### Error: "Too many values to unpack"
```python
# Wrong (4 items):
for x, y, x_mark, y_mark in loader:

# Correct (3 items):
for x, y, x_mark in loader:
```

### Error: "Model() takes 2 positional arguments but 4 were given"
```python
# Wrong (4 args):
outputs = model(x, x_mark, dec_inp, y_mark)

# Correct (2 args):
outputs = model(x, x_mark)
```

### Error: "Shape mismatch"
Check that `enc_in` and `c_out` match your dataset features (7 for ETT).

---

## 📚 Additional Resources

- **Paper**: [TimesNet (ICLR 2023)](https://openreview.net/pdf?id=ju_Uqw384Oq)
- **Original Code**: [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- **Dataset**: ETT (Electricity Transformer Temperature)

---

## ✅ Validation Checklist

Before training, verify:
- [ ] Dataset path is correct
- [ ] `enc_in` = `c_out` = number of features (7 for ETT)
- [ ] `seq_len` and `pred_len` are set
- [ ] Using paper hyperparameters (d_model, top_k, etc.)
- [ ] `scale=True` for normalization
- [ ] Only 3 items from dataset: `(x, y, x_mark)`
- [ ] Only 2 args to model: `model(x, x_mark)`

Run `python test_implementation.py` to auto-check! ✓

---

## 🎉 You're Ready!

1. **Quick Start**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **Understand Parameters**: See [PARAMETERS_GUIDE.md](PARAMETERS_GUIDE.md)
3. **Learn Architecture**: See [ARCHITECTURE_EXPLAINED.md](ARCHITECTURE_EXPLAINED.md)
4. **Train Your Model**: Run the examples above!

**Happy Forecasting!** 🚀

---

*This implementation is simplified and paper-faithful. All unnecessary parameters have been removed for clarity while maintaining full TimesNet functionality.*
