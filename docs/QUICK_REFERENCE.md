# TimesNet Quick Reference Card

## 🚀 Quick Start (Copy & Paste)

### Minimal Training Script
```python
from src.train import train_timesnet

config = {
    # Data
    'root_path': './ETDataset/ETT-small/',
    'data_path': 'ETTh1.csv',
    'seq_len': 96,
    'pred_len': 96,

    # Model (paper values for ETTh1)
    'enc_in': 7,
    'c_out': 7,
    'd_model': 16,
    'd_ff': 32,
    'top_k': 5,
    'e_layers': 2,
}

trainer, results = train_timesnet(config)
print(f"MSE: {results['mse']:.4f}, MAE: {results['mae']:.4f}")
```

### Manual Training Loop
```python
from src.dataset import create_dataloaders
from src.model import TimesNet
import torch.nn as nn
import torch.optim as optim

# Data
train_loader, val_loader, test_loader, dataset = create_dataloaders(
    root_path='./ETDataset/ETT-small/',
    data_path='ETTh1.csv',
    seq_len=96,
    pred_len=96,
    batch_size=32
)

# Model
model = TimesNet(seq_len=96, pred_len=96, enc_in=7, c_out=7, d_model=16, d_ff=32, top_k=5, e_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train
for epoch in range(10):
    for batch_x, batch_y, batch_x_mark in train_loader:  # 3 items!
        optimizer.zero_grad()
        outputs = model(batch_x, batch_x_mark)  # 2 args!
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

---

## 📋 Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_path` | str | - | Data directory path |
| `data_path` | str | 'ETTh1.csv' | CSV filename |
| `seq_len` | int | 96 | Input length |
| `pred_len` | int | 96 | Prediction horizon |
| `scale` | bool | True | Normalize data |
| `timeenc` | int | 0 | Time encoding: 0=manual, 1=Fourier |
| `freq` | str | 'h' | 'h'=hourly, 't'=15-min |

**Returns**: `(seq_x, seq_y, seq_x_mark)` - 3 items

---

## 🧠 Model Parameters

| Parameter | ETTh1 | ETTm1 | Description |
|-----------|-------|-------|-------------|
| `seq_len` | 96 | 96 | Input length |
| `pred_len` | 96 | 96 | Output length |
| `enc_in` | 7 | 7 | Input features |
| `c_out` | 7 | 7 | Output features |
| `d_model` | 16 | 32 | Model dimension |
| `d_ff` | 32 | 64 | FFN dimension |
| `top_k` | 5 | 5 | Top-k frequencies |
| `e_layers` | 2 | 2 | Number of TimesBlocks |
| `num_kernels` | 6 | 6 | Inception kernels |
| `dropout` | 0.1 | 0.1 | Dropout rate |
| `embed` | 'fixed' | 'fixed' | Embedding type |
| `freq` | 'h' | 't' | Time frequency |

**Forward**: `model(x_enc, x_mark_enc)` - 2 args

---

## ⚙️ Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_epochs` | 10 | Number of epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 0.0001 | Adam LR |
| `patience` | 3 | Early stopping patience |
| `lradj` | 'type1' | LR schedule |
| `use_amp` | False | Mixed precision |

---

## 📊 Common Prediction Horizons

### ETTh1 (hourly)
```python
# 96 → 96 (4 days)
config = {'seq_len': 96, 'pred_len': 96, 'd_model': 16, 'd_ff': 32}

# 96 → 192 (8 days)
config = {'seq_len': 96, 'pred_len': 192, 'd_model': 16, 'd_ff': 32}

# 96 → 336 (14 days)
config = {'seq_len': 96, 'pred_len': 336, 'd_model': 16, 'd_ff': 32}

# 96 → 720 (30 days)
config = {'seq_len': 96, 'pred_len': 720, 'd_model': 16, 'd_ff': 32}
```

### ETTm1 (15-minute)
```python
# 96 → 96 (24 hours)
config = {'seq_len': 96, 'pred_len': 96, 'd_model': 32, 'd_ff': 64}

# 96 → 192 (48 hours)
config = {'seq_len': 96, 'pred_len': 192, 'd_model': 32, 'd_ff': 64}
```

---

## 🔧 What `scale` and `timeenc` Do

### `scale=True` (Normalization)
```python
# Fit on train data only
scaler.fit(train_data)
# Transform all data
normalized = scaler.transform(all_data)

# Result: zero mean, unit variance
# Essential for training stability
```

### `timeenc=0` (Manual - Default)
```python
# Integer categorical encoding
month = 1-12
day = 1-31
weekday = 0-6
hour = 0-23

# Fed to embedding layers
```

### `timeenc=1` (Fourier)
```python
# Continuous normalized values
hour / 23.0 - 0.5        # [-0.5, 0.5]
dayofweek / 6.0 - 0.5    # [-0.5, 0.5]

# Fed to linear layer
```

**NOT the same as FFT in model!** That's for period detection.

---

## 🎯 Key Differences from Other Models

### ❌ Removed (not used by TimesNet):
- `label_len` - decoder overlap
- `features` - forecasting mode ('M'/'S'/'MS')
- `target` - target column
- `x_dec` - decoder input
- `x_mark_dec` - decoder time features

### ✅ TimesNet uses:
- **Encoder-only** architecture
- **FFT** to find dominant periods
- **2D convolutions** on reshaped time series
- **Direct projection** from input to output length

---

## 📝 Common Mistakes

### ❌ Wrong:
```python
# Old style (decoder-based models)
for x, y, x_mark, y_mark in loader:  # 4 items
    dec_inp = torch.zeros_like(y[:, -96:, :])
    dec_inp = torch.cat([y[:, :48, :], dec_inp], dim=1)
    outputs = model(x, x_mark, dec_inp, y_mark)  # 4 args
```

### ✅ Correct:
```python
# TimesNet style (encoder-only)
for x, y, x_mark in loader:  # 3 items
    outputs = model(x, x_mark)  # 2 args
```

---

## 🧪 Validation

Run this to verify your setup:
```bash
python test_implementation.py
```

Should see:
```
✓ Dataset created successfully
✓ DataLoaders created successfully
✓ Model created successfully
✓ Forward pass successful with 2 args
✓ All tests passed!
```

---

## 📚 Documentation Files

- `QUICK_REFERENCE.md` ← You are here
- `PARAMETERS_GUIDE.md` - Detailed parameter explanations
- `ARCHITECTURE_EXPLAINED.md` - Visual architecture guide
- `SIMPLIFIED_CHANGES.md` - What was changed and why
- `UPDATE_SUMMARY.md` - Complete update summary
- `test_implementation.py` - Validation script

---

## 🔗 Paper Reference

**TimesNet**: Temporal 2D-Variation Modeling for General Time Series Analysis
- Link: https://openreview.net/pdf?id=ju_Uqw384Oq
- Key innovation: FFT-based period detection + 2D convolutions
- SOTA on multiple forecasting benchmarks

---

## 💡 Pro Tips

1. **Start small**: Use `d_model=16, d_ff=32` for ETT datasets
2. **Use defaults**: Most parameters have good defaults
3. **Monitor early**: Use `patience=3` for early stopping
4. **Scale your data**: Always use `scale=True`
5. **Keep it simple**: Only configure `seq_len` and `pred_len` to start

---

## ⚡ One-Liner Examples

### Train ETTh1 96→96:
```python
from src.train import train_timesnet; trainer, results = train_timesnet({'root_path': './ETDataset/ETT-small/', 'data_path': 'ETTh1.csv', 'seq_len': 96, 'pred_len': 96, 'enc_in': 7, 'c_out': 7, 'd_model': 16, 'd_ff': 32, 'top_k': 5, 'e_layers': 2})
```

### Train ETTm1 96→96:
```python
from src.train import train_timesnet; trainer, results = train_timesnet({'root_path': './ETDataset/ETT-small/', 'data_path': 'ETTm1.csv', 'seq_len': 96, 'pred_len': 96, 'enc_in': 7, 'c_out': 7, 'd_model': 32, 'd_ff': 64, 'top_k': 5, 'e_layers': 2})
```

---

**Happy Forecasting! 🚀**
