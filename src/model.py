"""
TimesNet Model Implementation
Paper: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
https://openreview.net/pdf?id=ju_Uqw384Oq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math


# ========== Embedding Layers ==========

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


# ========== Convolution Blocks ==========

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


# ========== TimesNet Core ==========

def FFT_for_Period(x, k=2):
    """
    Find top-k periods using FFT

    Args:
        x: Input tensor [B, T, C]
        k: Number of top frequencies to extract

    Returns:
        period: Top-k periods [k]
        period_weight: Corresponding amplitudes [B, k]
    """
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # Find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # Parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # Padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(
                    x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # Reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # Reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # Adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # Residual connection
        res = res + x
        return res


# ========== TimesNet Model ==========

class TimesNet(nn.Module):
    """
    TimesNet for Long-term Time Series Forecasting

    Args:
        seq_len (int): Input sequence length
        pred_len (int): Prediction horizon
        enc_in (int): Number of input features
        c_out (int): Number of output features
        d_model (int): Model dimension (default: 64)
        d_ff (int): Feed-forward dimension (default: 128)
        num_kernels (int): Number of inception kernels (default: 6)
        top_k (int): Number of top frequencies (default: 5)
        e_layers (int): Number of encoder layers (default: 2)
        dropout (float): Dropout rate (default: 0.1)
        embed (str): Embedding type - 'fixed' or 'timeF' (default: 'fixed')
        freq (str): Time frequency - 'h' or 't' (default: 'h')
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        enc_in: int = 7,
        c_out: int = 7,
        d_model: int = 64,
        d_ff: int = 128,
        num_kernels: int = 6,
        top_k: int = 5,
        e_layers: int = 2,
        dropout: float = 0.1,
        embed: str = 'fixed',
        freq: str = 'h'
    ):
        super(TimesNet, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.e_layers = e_layers

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)

        # TimesBlocks
        self.model = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

        # Projection
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc=None):
        """
        Forward pass for forecasting

        Args:
            x_enc: Encoder input [batch, seq_len, features]
            x_mark_enc: Encoder time features [batch, seq_len, time_features] (optional)

        Returns:
            predictions: [batch, pred_len, features]
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # Align temporal dimension

        # TimesNet
        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        # Return only predictions
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, features]


def create_timesnet_model(config):
    """
    Create TimesNet model from configuration dictionary

    Args:
        config (dict): Configuration with keys:
            - seq_len, pred_len, enc_in, c_out
            - d_model, d_ff, num_kernels, top_k, e_layers
            - dropout, embed, freq

    Returns:
        model: TimesNet instance
    """
    return TimesNet(
        seq_len=config.get('seq_len', 96),
        pred_len=config.get('pred_len', 96),
        enc_in=config.get('enc_in', 7),
        c_out=config.get('c_out', 7),
        d_model=config.get('d_model', 64),
        d_ff=config.get('d_ff', 128),
        num_kernels=config.get('num_kernels', 6),
        top_k=config.get('top_k', 5),
        e_layers=config.get('e_layers', 2),
        dropout=config.get('dropout', 0.1),
        embed=config.get('embed', 'fixed'),
        freq=config.get('freq', 'h')
    )
