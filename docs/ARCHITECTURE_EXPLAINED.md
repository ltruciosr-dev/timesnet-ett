# TimesNet Architecture Explained

## 🎯 Core Understanding

### What TimesNet IS:
```
┌─────────────────────────────────────────────────────────┐
│                    ENCODER-ONLY MODEL                    │
│                                                          │
│  Input → Embedding → TimesBlocks → Linear → Predictions │
│                         (FFT)                            │
└─────────────────────────────────────────────────────────┘
```
---

## 📊 Data Flow Diagram

```
INPUT DATA (CSV)
    │
    ├─── date: 2016-07-01 00:00:00
    ├─── HUFL: 5.827
    ├─── HULL: 2.009
    ├─── MUFL: 1.599
    ├─── MULL: 0.462
    ├─── LUFL: 4.203
    ├─── LULL: 1.340
    └─── OT: 30.531

         ↓

DATASET PROCESSING (dataset.py)
    │
    ├─── StandardScaler (if scale=True)
    │    ├─── Fit on train data only
    │    └─── Transform all data
    │
    ├─── Time Features (timeenc)
    │    ├─── Manual (0): [month, day, weekday, hour]
    │    └─── Fourier (1): [normalized temporal values]
    │
    └─── Train/Val/Test Split
         ├─── Train: 70%
         ├─── Val: 10%
         └─── Test: 20%

         ↓

DATASET __getitem__ RETURNS
    │
    ├─── seq_x: [seq_len=96, features=7]       # Input sequence
    ├─── seq_y: [pred_len=96, features=7]      # Target (future values)
    └─── seq_x_mark: [seq_len=96, time_feat=4] # Time features

    ❌ NO seq_y_mark (removed!)

         ↓

DATALOADER (batch processing)
    │
    └─── Batches: [batch_size=32, seq_len=96, features=7]

         ↓

MODEL FORWARD PASS (model.py)
    │
    ├─── Input: batch_x [B, 96, 7], batch_x_mark [B, 96, 4]
    │
    ├─── 1. NORMALIZATION (Non-stationary)
    │    ├─── means = x.mean(dim=1)
    │    ├─── stdev = x.std(dim=1)
    │    └─── x_normalized = (x - means) / stdev
    │
    ├─── 2. EMBEDDING
    │    ├─── Value Embedding (1D Conv): [B, 96, 7] → [B, 96, 16]
    │    ├─── Temporal Embedding: [B, 96, 4] → [B, 96, 16]
    │    └─── Positional Embedding: [B, 96, 16]
    │    └─── Combined: [B, 96, 16]
    │
    ├─── 3. LINEAR PROJECTION
    │    └─── [B, 96, 16] → [B, 144, 16]  (96+96=144)
    │
    ├─── 4. TIMESBLOCKS (×2)
    │    │
    │    ├─── A. FFT_for_Period (find top-k=5 frequencies)
    │    │    ├─── FFT: [B, 144, 16] → frequency domain
    │    │    ├─── Amplitude: |FFT|.mean(0).mean(-1)
    │    │    └─── Top-k: argmax(amplitudes, k=5) → periods
    │    │
    │    ├─── B. Reshape to 2D
    │    │    └─── [B, 144, 16] → [B, 16, period, 144/period]
    │    │
    │    ├─── C. Inception_Block_V1 (multi-scale 2D Conv)
    │    │    ├─── Kernel 1×1: [B, 16, *, *] → [B, 32, *, *]
    │    │    ├─── Kernel 3×3: [B, 16, *, *] → [B, 32, *, *]
    │    │    ├─── Kernel 5×5: [B, 16, *, *] → [B, 32, *, *]
    │    │    ├─── ... (6 kernels total)
    │    │    └─── Average: [B, 32, *, *]
    │    │
    │    ├─── D. GELU activation
    │    │
    │    ├─── E. Second Inception Block
    │    │    └─── [B, 32, *, *] → [B, 16, *, *]
    │    │
    │    ├─── F. Reshape back to 1D
    │    │    └─── [B, 16, *, *] → [B, 144, 16]
    │    │
    │    ├─── G. Weight by amplitudes (adaptive aggregation)
    │    │    └─── softmax(amplitudes) * outputs
    │    │
    │    └─── H. Residual connection
    │         └─── output = weighted_sum + input
    │
    ├─── 5. LAYER NORM
    │    └─── [B, 144, 16] → normalized
    │
    ├─── 6. PROJECTION
    │    └─── [B, 144, 16] → [B, 144, 7]
    │
    ├─── 7. DE-NORMALIZATION
    │    └─── output * stdev + means
    │
    └─── 8. EXTRACT PREDICTIONS
         └─── [B, 144, 7][-96:] → [B, 96, 7]

         ↓

OUTPUT
    └─── Predictions: [batch_size=32, pred_len=96, features=7]
```

---

## 🔧 Two Different FFT Uses

### 1. Time Encoding (Optional, in dataset.py)
```python
if timeenc == 1:  # Fourier time features
    # Encodes calendar information as continuous values
    hour / 23.0 - 0.5
    dayofweek / 6.0 - 0.5
    # etc.
```
**Purpose**: Tell model "what time it is"
**Used**: Optionally for time features
**Paper uses**: `timeenc=0` (manual encoding)

### 2. Period Detection (Always, in model.py)
```python
def FFT_for_Period(x, k=5):  # TimesNet core!
    xf = torch.fft.rfft(x, dim=1)
    amplitudes = abs(xf).mean(0).mean(-1)
    top_frequencies = topk(amplitudes, k)
    periods = seq_len // top_frequencies
```
**Purpose**: Find dominant periods (daily, weekly, etc.)
**Used**: Always, with top_k=5
**This is TimesNet's innovation!**

---

## 📈 Example: ETTh1 96→96 Prediction

### Input:
```
Sequence: [2016-07-01 00:00, ..., 2016-07-04 23:00]  # 96 hours
Features: [HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]   # 7 features
```

### Processing:
1. **Normalize**: zero mean, unit variance
2. **Embed**: 7 features → 16 dimensions
3. **Extend**: 96 timesteps → 144 timesteps (96+96)
4. **FFT**: Find top-5 periods (e.g., 24h, 12h, 8h, 6h, 4h)
5. **2D Conv**: Multi-scale pattern extraction for each period
6. **Aggregate**: Weighted sum based on period importance
7. **Project**: 16 dimensions → 7 features
8. **Extract**: Last 96 timesteps

### Output:
```
Predictions: [2016-07-05 00:00, ..., 2016-07-08 23:00]  # 96 hours
Features: [HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]      # 7 features
```

---

## 🔑 Key Design Choices

### 1. **Why no decoder?**
- TimesNet uses direct projection: seq_len → seq_len+pred_len
- Then extracts last pred_len steps
- Simpler and more efficient than autoregressive decoding

### 2. **Why 2D convolutions?**
- Reshape 1D time series based on discovered periods
- Vertical axis = period, Horizontal axis = cycles
- 2D patterns capture both intra-period and inter-period variations

### 3. **Why multiple periods (top-k)?**
- Real data has multiple periodicities (hourly, daily, weekly)
- Different periods capture different patterns
- Adaptive aggregation weights them by importance

### 4. **Why Inception blocks?**
- Multi-scale kernels (1×1, 3×3, 5×5, ..., 11×11)
- Capture both fine and coarse patterns
- Parameter-efficient (shared input)

---

## 🎓 Paper Values Reference

### ETTh1 (96 → 96)
```python
# Data
seq_len = 96       # 4 days of hourly data
pred_len = 96      # Predict next 4 days
scale = True       # Normalize
timeenc = 0        # Manual time encoding

# Model
enc_in = 7         # 7 features
c_out = 7
d_model = 16       # Small for ETT
d_ff = 32
top_k = 5          # Top-5 periods
e_layers = 2       # 2 TimesBlocks
num_kernels = 6    # 6 inception kernels
dropout = 0.1
embed = 'fixed'    # Fixed embeddings

# Training
lr = 0.0001        # Adam
batch_size = 32
epochs = 10
patience = 3       # Early stopping
```

### Other Horizons (ETTh1)
```python
96 → 192:  same params, just pred_len=192
96 → 336:  same params, just pred_len=336
96 → 720:  same params, just pred_len=720
```

---

## ✅ Implementation Checklist

- [x] Dataset returns 3 items (x, y, x_mark)
- [x] Model takes 2 arguments (x_enc, x_mark_enc)
- [x] No decoder inputs or label_len
- [x] FFT finds top-k periods in model
- [x] 2D convolutions for multi-period modeling
- [x] Adaptive aggregation by amplitude
- [x] Non-stationary normalization
- [x] Paper hyperparameters (d_model=16, top_k=5, etc.)

---

## 🚀 That's TimesNet!

**Core Innovation**: Transform 1D time series to 2D based on discovered periods, then use 2D convolutions to capture complex temporal patterns.

**Key Advantage**: No need for:
- ❌ Complex attention mechanisms
- ❌ Decoder architectures
- ❌ Autoregressive generation
- ❌ Position-dependent operations

**Result**: Simple, efficient, and effective! 🎉
