"""
Training Script for TimesNet on ETT Dataset
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from dataset import create_dataloaders
from model import create_timesnet_model
from evaluate import evaluate_model


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, learning_rate, lradj='type1'):
    """Adjust learning rate based on schedule"""
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj == 'constant':
        lr_adjust = {epoch: learning_rate}
    else:
        lr_adjust = {epoch: learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


class TimesNetTrainer:
    """
    TimesNet Trainer for ETT Dataset

    Args:
        config (dict): Configuration dictionary with keys:
            Data:
                - root_path: Path to data directory
                - data_path: CSV filename (e.g., 'ETTh1.csv')
                - seq_len: Input sequence length (default: 96)
                - pred_len: Prediction horizon (default: 96)

            Model:
                - enc_in: Input features (default: 7)
                - c_out: Output features (default: 7)
                - d_model: Model dimension (default: 64)
                - d_ff: Feed-forward dimension (default: 128)
                - num_kernels: Inception kernels (default: 6)
                - top_k: Top frequencies (default: 5)
                - e_layers: Encoder layers (default: 2)
                - dropout: Dropout rate (default: 0.1)
                - embed: Embedding type (default: 'fixed')

            Training:
                - train_epochs: Number of epochs (default: 10)
                - batch_size: Batch size (default: 32)
                - learning_rate: Initial LR (default: 0.0001)
                - patience: Early stopping patience (default: 3)
                - lradj: LR schedule type (default: 'type1')
                - use_amp: Use automatic mixed precision (default: False)
                - num_workers: Data loading workers (default: 0)

            System:
                - device: 'cuda' or 'cpu' (default: auto-detect)
                - checkpoints: Checkpoint directory (default: './checkpoints')
    """

    def __init__(self, config):
        self.config = config

        # Set device
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Create checkpoints directory
        self.checkpoint_dir = config.get('checkpoints', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize data loaders
        self._init_data()

        # Initialize model
        self._init_model()

    def _init_data(self):
        """Initialize data loaders"""
        print('Initializing data loaders...')
        self.train_loader, self.val_loader, self.test_loader, self.train_dataset = create_dataloaders(
            root_path=self.config['root_path'],
            data_path=self.config['data_path'],
            batch_size=self.config.get('batch_size', 32),
            seq_len=self.config.get('seq_len', 96),
            pred_len=self.config.get('pred_len', 96),
            num_workers=self.config.get('num_workers', 0)
        )
        print(f'Train samples: {len(self.train_loader.dataset)}')
        print(f'Val samples: {len(self.val_loader.dataset)}')
        print(f'Test samples: {len(self.test_loader.dataset)}')

    def _init_model(self):
        """Initialize model, optimizer, and loss"""
        print('Initializing model...')

        # Determine frequency
        freq = 't' if 'ETTm' in self.config['data_path'] else 'h'

        # Create model config
        model_config = {
            'seq_len': self.config.get('seq_len', 96),
            'pred_len': self.config.get('pred_len', 96),
            'enc_in': self.config.get('enc_in', 7),
            'c_out': self.config.get('c_out', 7),
            'd_model': self.config.get('d_model', 64),
            'd_ff': self.config.get('d_ff', 128),
            'num_kernels': self.config.get('num_kernels', 6),
            'top_k': self.config.get('top_k', 5),
            'e_layers': self.config.get('e_layers', 2),
            'dropout': self.config.get('dropout', 0.1),
            'embed': self.config.get('embed', 'fixed'),
            'freq': freq
        }

        self.model = create_timesnet_model(model_config).to(self.device)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model parameters: {num_params:,}')

        # Optimizer (Adam as used in Time-Series-Library)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.0001)
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # AMP scaler for mixed precision training
        self.use_amp = self.config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark):
        """Process one batch"""
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        # Forward pass (TimesNet doesn't use decoder inputs)
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark)

        return outputs, batch_y

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        train_loss = []

        for i, (batch_x, batch_y, batch_x_mark) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            outputs, batch_y = self._process_one_batch(batch_x, batch_y, batch_x_mark)

            loss = self.criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print(f'\tIter: {i + 1}, Loss: {loss.item():.7f}')

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return np.average(train_loss)

    def validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(self.val_loader):
                outputs, batch_y = self._process_one_batch(batch_x, batch_y, batch_x_mark)
                loss = self.criterion(outputs, batch_y)
                val_loss.append(loss.item())

        return np.average(val_loss)

    def train(self):
        """Full training loop"""
        print('=' * 50)
        print('Starting training...')
        print('=' * 50)

        train_epochs = self.config.get('train_epochs', 10)
        patience = self.config.get('patience', 3)
        lradj = self.config.get('lradj', 'type1')

        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        train_losses = []
        val_losses = []

        for epoch in range(train_epochs):
            epoch_time = time.time()

            train_loss = self.train_epoch(epoch + 1)
            val_loss = self.validate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch: {epoch + 1} | Time: {time.time() - epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}')

            # Early stopping
            early_stopping(val_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print('Early stopping triggered')
                break

            # Adjust learning rate
            adjust_learning_rate(self.optimizer, epoch + 1, self.config.get('learning_rate', 0.0001), lradj)

        # Load best model
        self.model.load_state_dict(torch.load(checkpoint_path))
        print('Training completed!')

        return train_losses, val_losses

    def test(self):
        """Test model"""
        print('=' * 50)
        print('Testing model...')
        print('=' * 50)

        # Load best model
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))
            print(f'Loaded best model from {checkpoint_path}')

        # Evaluate
        results = evaluate_model(
            self.model, self.test_loader, self.device,
            self.train_dataset.scaler
        )

        print(f"Test MSE: {results['mse']:.7f}")
        print(f"Test MAE: {results['mae']:.7f}")
        print(f"Test RMSE: {results['rmse']:.7f}")

        return results


def train_timesnet(config):
    """
    Convenience function to train TimesNet

    Args:
        config (dict): Training configuration

    Returns:
        trainer: TimesNetTrainer instance
        results: Test results
    """
    trainer = TimesNetTrainer(config)
    train_losses, val_losses = trainer.train()
    results = trainer.test()

    return trainer, results


if __name__ == '__main__':
    # Example configuration for ETTh1
    config = {
        # Data
        'root_path': '/Users/ltruciosr/Documents/utec/deep_learning/project_2/ETDataset/ETT-small',
        'data_path': 'ETTh1.csv',
        'seq_len': 96,
        'pred_len': 24,

        # Model (paper hyperparameters for ETTh1)
        'enc_in': 7,
        'c_out': 7,
        'd_model': 16,
        'd_ff': 32,
        'num_kernels': 6,
        'top_k': 5,
        'e_layers': 2,
        'dropout': 0.1,
        'embed': 'fixed',

        # Training
        'train_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'patience': 3,
        'lradj': 'type1',
        'use_amp': False,
        'num_workers': 0,

        # System
        'checkpoints': './checkpoints',
    }

    # Train
    trainer, results = train_timesnet(config)
    print('\nTraining completed!')
    print(f"Final Test MSE: {results['mse']:.7f}")
    print(f"Final Test MAE: {results['mae']:.7f}")
