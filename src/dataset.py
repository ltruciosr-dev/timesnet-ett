"""
ETT Dataset for Time Series Forecasting
Supports ETTh1, ETTh2, ETTm1, ETTm2 datasets
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class ETTDataset(Dataset):
    """
    ETT (Electricity Transformer Temperature) Dataset for TimesNet forecasting

    Args:
        root_path (str): Root directory containing CSV files
        data_path (str): CSV filename (e.g., 'ETTh1.csv')
        flag (str): Split type - 'train', 'val', or 'test'
        seq_len (int): Input sequence length (default: 96)
        pred_len (int): Prediction horizon (default: 96)
        scale (bool): Whether to normalize data (default: True)
        timeenc (int): Time encoding mode - 0 (manual), 1 (Fourier features)
        freq (str): Time frequency - 'h' (hourly), 't' (15-minute)
    """
    def __init__(
        self,
        root_path: str,
        data_path: str = 'ETTh1.csv',
        flag: str = 'train',
        seq_len: int = 96,
        pred_len: int = 96,
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h'
    ):
        # Sequence configuration
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Split type
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Configuration
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # Paths
        self.root_path = root_path
        self.data_path = data_path

        # Load and preprocess data
        self.__read_data__()

    def __read_data__(self):
        """Load CSV and preprocess data"""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Determine borders for train/val/test splits using 70%/10%/20% ratio
        total_length = len(df_raw)
        num_train = int(total_length * 0.7)
        num_val = int(total_length * 0.1)
        num_test = total_length - num_train - num_val

        border1s = [0, num_train - self.seq_len, num_train + num_val - self.seq_len]
        border2s = [num_train, num_train + num_val, num_train + num_val + num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Use all features except 'date' (multivariate forecasting)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        # Normalization
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Time features encoding
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            # Manual encoding
            df_stamp['month'] = df_stamp.date.apply(lambda x: x.month)
            df_stamp['day'] = df_stamp.date.apply(lambda x: x.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda x: x.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda x: x.hour)
            if self.freq == 't':
                df_stamp['minute'] = df_stamp.date.apply(lambda x: x.minute)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # Fourier time features
            data_stamp = self._time_features(df_stamp['date'].values, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # Store processed data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def _time_features(self, dates, freq='h'):
        """Generate Fourier time features"""
        dates = pd.to_datetime(dates)

        if freq == 'h':
            # Hourly: hour, day of week, day of month, day of year
            features = np.vstack([
                dates.hour / 23.0 - 0.5,
                dates.dayofweek / 6.0 - 0.5,
                (dates.day - 1) / 30.0 - 0.5,
                (dates.dayofyear - 1) / 365.0 - 0.5
            ])
        elif freq == 't':
            # 15-minute: minute, hour, day of week, day of month, day of year
            features = np.vstack([
                dates.minute / 59.0 - 0.5,
                dates.hour / 23.0 - 0.5,
                dates.dayofweek / 6.0 - 0.5,
                (dates.day - 1) / 30.0 - 0.5,
                (dates.dayofyear - 1) / 365.0 - 0.5
            ])
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

        return features.astype(np.float32)

    def __getitem__(self, index):
        """
        Returns:
            seq_x: Input sequence [seq_len, features]
            seq_y: Target sequence [pred_len, features]
            seq_x_mark: Input time features [seq_len, time_features]

        Note: TimesNet encoder-only uses ONLY input time features for embedding.
        The model learns to predict future values without future time features.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        y_begin = s_end
        y_end = y_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[y_begin:y_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]  # Only input time features

        return seq_x, seq_y, seq_x_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse normalization for predictions"""
        return self.scaler.inverse_transform(data)


def create_dataloaders(
    root_path: str,
    data_path: str = 'ETTh1.csv',
    batch_size: int = 32,
    seq_len: int = 96,
    pred_len: int = 96,
    num_workers: int = 0
):
    """
    Create train/val/test dataloaders for ETT dataset

    Args:
        root_path (str): Root directory containing CSV files
        data_path (str): CSV filename
        batch_size (int): Batch size
        seq_len (int): Input sequence length
        pred_len (int): Prediction horizon
        num_workers (int): Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader, train_dataset (for scaler access)
    """
    from torch.utils.data import DataLoader

    # Determine frequency from filename
    freq = 't' if 'ETTm' in data_path else 'h'

    # Create datasets for train, validation, and test splits
    train_dataset = ETTDataset(
        root_path=root_path,
        data_path=data_path,
        flag='train',
        seq_len=seq_len,
        pred_len=pred_len,
        freq=freq
    )

    val_dataset = ETTDataset(
        root_path=root_path,
        data_path=data_path,
        flag='val',
        seq_len=seq_len,
        pred_len=pred_len,
        freq=freq
    )

    test_dataset = ETTDataset(
        root_path=root_path,
        data_path=data_path,
        flag='test',
        seq_len=seq_len,
        pred_len=pred_len,
        freq=freq
    )

    # Create data loaders with appropriate configurations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data for better generalization
        num_workers=num_workers,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation to maintain consistent evaluation
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for test to maintain consistent evaluation
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, train_dataset
