# TimesNet Parameters Guide

## Quick Reference: What's Actually Used

### ✅ Dataset Parameters (dataset.py)

| Parameter | Type | Default | Used? | Description |
|-----------|------|---------|-------|-------------|
| `root_path` | str | - | ✅ | Path to data directory |
| `data_path` | str | 'ETTh1.csv' | ✅ | CSV filename |
| `flag` | str | 'train' | ✅ | Split: 'train'/'val'/'test' |
| `seq_len` | int | 96 | ✅ | Input sequence length |
| `pred_len` | int | 96 | ✅ | Prediction horizon |
| `scale` | bool | True | ✅ | Normalize data with StandardScaler |
| `timeenc` | int | 0 | ✅ | Time encoding: 0=manual, 1=Fourier |
| `freq` | str | 'h' | ✅ | Frequency: 'h'=hourly, 't'=15-min |
| ~~`label_len`~~ | ~~int~~ | ~~48~~ | ❌ | **REMOVED** - Not used by TimesNet |
| ~~`features`~~ | ~~str~~ | ~~'M'~~ | ❌ | **REMOVED** - Always multivariate |
| ~~`target`~~ | ~~str~~ | ~~'OT'~~ | ❌ | **REMOVED** - Not used |

### ✅ Model Parameters (model.py)

| Parameter | Type | Default | Used? | Description |
|-----------|------|---------|-------|-------------|
| `seq_len` | int | 96 | ✅ | Input sequence length |
| `pred_len` | int | 96 | ✅ | Prediction horizon |
| `enc_in` | int | 7 | ✅ | Number of input features |
| `c_out` | int | 7 | ✅ | Number of output features |
| `d_model` | int | 64 | ✅ | Model dimension |
| `d_ff` | int | 128 | ✅ | Feed-forward dimension |
| `num_kernels` | int | 6 | ✅ | Inception block kernels |
| `top_k` | int | 5 | ✅ | Top-k frequencies for FFT |
| `e_layers` | int | 2 | ✅ | Number of TimesBlocks |
| `dropout` | float | 0.1 | ✅ | Dropout rate |
| `embed` | str | 'fixed' | ✅ | Embedding type: 'fixed'/'timeF' |
| `freq` | str | 'h' | ✅ | Time frequency for embeddings |

### ✅ Training Parameters (train.py)

| Parameter | Type | Default | Used? | Description |
|-----------|------|---------|-------|-------------|
| `train_epochs` | int | 10 | ✅ | Number of training epochs |
| `batch_size` | int | 32 | ✅ | Batch size |
| `learning_rate` | float | 0.0001 | ✅ | Initial learning rate |
| `patience` | int | 3 | ✅ | Early stopping patience |
| `lradj` | str | 'type1' | ✅ | LR schedule type |
| `use_amp` | bool | False | ✅ | Use mixed precision |
| `num_workers` | int | 0 | ✅ | DataLoader workers |
| `checkpoints` | str | './checkpoints' | ✅ | Checkpoint directory |
| `device` | str | auto | ✅ | 'cuda' or 'cpu' |

---

## Paper Values (ETTh1 96→96)

### Dataset
```python
seq_len = 96
pred_len = 96
scale = True
timeenc = 0      # Manual time encoding
freq = 'h'       # Hourly data
```

### Model
```python
enc_in = 7       # 7 features in ETT
c_out = 7
d_model = 16     # Small model for ETT
d_ff = 32
num_kernels = 6
top_k = 5        # Top-5 frequencies
e_layers = 2
dropout = 0.1
embed = 'fixed'  # Fixed embeddings
```

### Training
```python
train_epochs = 10
batch_size = 32
learning_rate = 0.0001
patience = 3
lradj = 'type1'  # Exponential decay
```

---

## Parameter Explanations

### 🔧 `scale` (Normalization)
**What it does**: Normalizes all features to zero mean and unit variance
```python
# Fit scaler on training data
scaler.fit(train_data)
# Transform all data
normalized_data = scaler.transform(data)
```

**Why**:
- Different features have different scales (temperature vs load)
- Helps neural network training stability
- Standard practice in deep learning

**Paper uses**: `True` ✅

---

### 🔧 `timeenc` (Time Feature Encoding)

#### Option 0: Manual Encoding (Default)
```python
# Categorical integers fed to embedding layers
month = 1-12
day = 1-31
weekday = 0-6
hour = 0-23
minute = 0-3 (for 15-min data, quarter-hour)
```

#### Option 1: Fourier Encoding
```python
# Continuous normalized values fed to linear layer
hour / 23.0 - 0.5          # [-0.5, 0.5]
dayofweek / 6.0 - 0.5      # [-0.5, 0.5]
(day - 1) / 30.0 - 0.5     # [-0.5, 0.5]
(dayofyear - 1) / 365.0 - 0.5  # [-0.5, 0.5]
```

**Purpose**: Tells the model "what time it is" (hour of day, day of week, etc.)

**NOT the same as**: FFT period detection in the model (that's always used with `top_k`)

**Paper uses**: `0` (manual encoding) ✅

---

### 🔧 `embed` (Embedding Type)

#### Option 1: 'fixed' (Default)
- Uses **FixedEmbedding** with sinusoidal patterns (non-trainable)
- For temporal features: fixed sinusoidal embeddings

#### Option 2: 'timeF'
- Uses **TimeFeatureEmbedding** with linear projection
- Requires `timeenc=1` (Fourier time features)

**Relationship**:
- `embed='fixed'` + `timeenc=0`: Manual time → Fixed embeddings ✅ (Paper default)
- `embed='timeF'` + `timeenc=1`: Fourier time → Linear projection

**Paper uses**: `'fixed'` ✅

---

### 🔧 `top_k` (FFT Period Selection)

**What it does**: In the model's `FFT_for_Period()` function
```python
xf = torch.fft.rfft(x, dim=1)           # FFT on time dimension
frequency_list = abs(xf).mean(0).mean(-1)  # Amplitude spectrum
_, top_list = torch.topk(frequency_list, k)  # Select top-k
period = seq_len // top_list             # Convert to periods
```

**Purpose**:
- Finds the **k most dominant periods** in the data
- Uses their amplitudes to weight multi-period representations
- Core innovation of TimesNet!

**Paper uses**: `5` ✅

---

### 🔧 `d_model` and `d_ff` (Model Size)

**ETTh1 values**: `d_model=16`, `d_ff=32` (very small!)

**Why so small?**
- ETT is a relatively simple dataset
- Paper shows smaller models work well for ETT
- Prevents overfitting on small datasets

**Other datasets use larger values**:
- Electricity: `d_model=32`, `d_ff=32`
- Weather: `d_model=32`, `d_ff=32`
- Traffic: `d_model=64`, `d_ff=64`

---

## Minimal Config Example

```python
# Absolute minimum for ETTh1
config = {
    # Data
    'root_path': './ETDataset/ETT-small/',
    'data_path': 'ETTh1.csv',
    'seq_len': 96,
    'pred_len': 96,

    # Model (paper values)
    'enc_in': 7,
    'c_out': 7,
    'd_model': 16,
    'd_ff': 32,
    'top_k': 5,
    'e_layers': 2,

    # Training
    'batch_size': 32,
    'learning_rate': 0.0001,
    'train_epochs': 10,
}
```

Everything else uses defaults! ✅

---

## Common Mistakes to Avoid

❌ **Don't do this:**
```python
# These parameters were removed!
config = {
    'label_len': 48,      # ❌ Not used by TimesNet
    'features': 'M',      # ❌ Always multivariate now
    'target': 'OT',       # ❌ Not needed
}
```

✅ **Do this instead:**
```python
config = {
    'seq_len': 96,
    'pred_len': 96,
    # That's it for sequence config!
}
```

---

## Dataset Returns (Important!)

### Old (removed):
```python
seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[i]  # 4 items
outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 4 args
```

### New (current):
```python
seq_x, seq_y, seq_x_mark = dataset[i]  # 3 items
outputs = model(batch_x, batch_x_mark)  # 2 args
```

**Why**: TimesNet doesn't use a decoder, so no decoder inputs/time features needed!

---

## Summary

### Always Required:
1. `seq_len` - input sequence length
2. `pred_len` - prediction horizon
3. `enc_in` / `c_out` - number of features
4. `root_path` / `data_path` - where data is

### Paper Defaults (ETTh1):
- `d_model=16`, `d_ff=32` (small model)
- `top_k=5` (top-5 frequencies)
- `e_layers=2` (2 TimesBlocks)
- `scale=True` (normalize)
- `timeenc=0` (manual time encoding)
- `embed='fixed'` (fixed embeddings)

### Removed Parameters:
- ❌ `label_len` - decoder overlap (not needed)
- ❌ `features` - forecasting mode (always multivariate)
- ❌ `target` - target column (not used)

**Result**: Simpler, cleaner, and matches the paper exactly! 🎉
