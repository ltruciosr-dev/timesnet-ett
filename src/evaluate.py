"""
Evaluation Script for TimesNet
Computes MSE, MAE, RMSE, MAPE, and MSPE metrics
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error


def RSE(pred, true):
    """Root Relative Squared Error"""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """Correlation coefficient"""
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """Mean Squared Error"""
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """Mean Squared Percentage Error"""
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    """
    Compute all metrics

    Args:
        pred: Predictions array [samples, features]
        true: Ground truth array [samples, features]

    Returns:
        dict: Dictionary with all metrics
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'mspe': mspe,
        'rse': rse,
        'corr': corr
    }


def evaluate_model(model, data_loader, device, scaler=None):
    """
    Evaluate model on a dataset

    Args:
        model: TimesNet model
        data_loader: PyTorch DataLoader
        device: 'cuda' or 'cpu'
        scaler: StandardScaler for inverse transform (optional)

    Returns:
        dict: Metrics dictionary
    """
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            # Forward pass
            outputs = model(batch_x, batch_x_mark)

            # Extract predictions and targets
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            preds.append(outputs)
            trues.append(batch_y)

    # Concatenate all batches
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Inverse transform if scaler provided
    if scaler is not None:
        # Reshape for inverse transform
        preds_shape = preds.shape
        trues_shape = trues.shape

        preds = scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds_shape)
        trues = scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues_shape)

    # Compute metrics
    results = metric(preds, trues)

    return results


def evaluate_and_save_results(model, train_loader, val_loader, test_loader, device, scaler, save_path=None):
    """
    Evaluate model on train/val/test sets and optionally save results

    Args:
        model: TimesNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: 'cuda' or 'cpu'
        scaler: StandardScaler for inverse transform
        save_path: Path to save results (optional)

    Returns:
        dict: Results for all splits
    """
    print('=' * 50)
    print('Evaluating model on all splits...')
    print('=' * 50)

    # Evaluate on all splits
    train_results = evaluate_model(model, train_loader, device, scaler)
    val_results = evaluate_model(model, val_loader, device, scaler)
    test_results = evaluate_model(model, test_loader, device, scaler)

    # Print results
    print('\nTrain Results:')
    print(f"  MSE: {train_results['mse']:.7f}")
    print(f"  MAE: {train_results['mae']:.7f}")
    print(f"  RMSE: {train_results['rmse']:.7f}")
    print(f"  MAPE: {train_results['mape']:.7f}")
    print(f"  MSPE: {train_results['mspe']:.7f}")

    print('\nValidation Results:')
    print(f"  MSE: {val_results['mse']:.7f}")
    print(f"  MAE: {val_results['mae']:.7f}")
    print(f"  RMSE: {val_results['rmse']:.7f}")
    print(f"  MAPE: {val_results['mape']:.7f}")
    print(f"  MSPE: {val_results['mspe']:.7f}")

    print('\nTest Results:')
    print(f"  MSE: {test_results['mse']:.7f}")
    print(f"  MAE: {test_results['mae']:.7f}")
    print(f"  RMSE: {test_results['rmse']:.7f}")
    print(f"  MAPE: {test_results['mape']:.7f}")
    print(f"  MSPE: {test_results['mspe']:.7f}")

    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write('Train Results:\n')
            for k, v in train_results.items():
                f.write(f'  {k}: {v:.7f}\n')

            f.write('\nValidation Results:\n')
            for k, v in val_results.items():
                f.write(f'  {k}: {v:.7f}\n')

            f.write('\nTest Results:\n')
            for k, v in test_results.items():
                f.write(f'  {k}: {v:.7f}\n')

        print(f'\nResults saved to {save_path}')

    return {
        'train': train_results,
        'val': val_results,
        'test': test_results
    }


def visualize_predictions(model, data_loader, device, scaler, num_samples=5, save_path=None):
    """
    Visualize predictions vs ground truth

    Args:
        model: TimesNet model
        data_loader: DataLoader
        device: 'cuda' or 'cpu'
        scaler: StandardScaler for inverse transform
        num_samples: Number of samples to visualize
        save_path: Path to save figure (optional)
    """
    model.eval()
    samples_plotted = 0

    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark in data_loader:
            if samples_plotted >= num_samples:
                break

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            # Forward pass
            outputs = model(batch_x, batch_x_mark)

            # Get predictions and targets
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            # Inverse transform
            if scaler is not None:
                outputs = scaler.inverse_transform(outputs[0])
                batch_y = scaler.inverse_transform(batch_y[0])
            else:
                outputs = outputs[0]
                batch_y = batch_y[0]

            # Plot first feature (typically 'OT')
            ax = axes[samples_plotted]
            ax.plot(batch_y[:, 0], label='Ground Truth', linewidth=2)
            ax.plot(outputs[:, 0], label='Prediction', linewidth=2, linestyle='--')
            ax.set_title(f'Sample {samples_plotted + 1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

            samples_plotted += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Predictions plot saved to {save_path}')

    plt.show()


def visualize_horizon_predictions(model, train_loader, val_loader, test_loader, device, scaler,
                                   seq_len, pred_len, num_samples=3, feature_idx=0,
                                   feature_name='OT', save_path=None):
    """
    Visualize predictions with input context showing the forecast horizon.
    Shows samples from train, validation, and test sets side by side.

    Args:
        model: TimesNet model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        device: 'cuda' or 'cpu'
        scaler: StandardScaler for inverse transform
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        num_samples: Number of samples to visualize per split
        feature_idx: Which feature to plot (default 0 for 'OT')
        feature_name: Name of the feature for plot labels
        save_path: Path to save figure (optional)
    """
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    loaders = [train_loader, val_loader, test_loader]
    split_names = ['Train', 'Validation', 'Test']

    with torch.no_grad():
        for col_idx, (loader, split_name) in enumerate(zip(loaders, split_names)):
            samples_plotted = 0

            for batch_x, batch_y, batch_x_mark in loader:
                if samples_plotted >= num_samples:
                    break

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)

                # Forward pass
                outputs = model(batch_x, batch_x_mark)

                # Get first sample from batch
                input_seq = batch_x[0].detach().cpu().numpy()
                pred = outputs[0].detach().cpu().numpy()
                true = batch_y[0].detach().cpu().numpy()

                # Inverse transform
                if scaler is not None:
                    input_seq = scaler.inverse_transform(input_seq)
                    pred = scaler.inverse_transform(pred)
                    true = scaler.inverse_transform(true)

                # Create time axis
                input_time = np.arange(seq_len)
                pred_time = np.arange(seq_len, seq_len + pred_len)

                # Plot
                ax = axes[samples_plotted, col_idx]

                # Input sequence (context)
                ax.plot(input_time, input_seq[:, feature_idx],
                       label='Input Context', color='gray', linewidth=2, alpha=0.7)

                # Ground truth forecast
                ax.plot(pred_time, true[:, feature_idx],
                       label='Ground Truth', color='green', linewidth=2.5)

                # Model prediction
                ax.plot(pred_time, pred[:, feature_idx],
                       label='Prediction', color='red', linewidth=2, linestyle='--')

                # Vertical line separating context and forecast
                ax.axvline(x=seq_len, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

                # Labels and formatting
                if samples_plotted == 0:
                    ax.set_title(f'{split_name} Split', fontsize=14, fontweight='bold')

                ax.set_xlabel('Time Step', fontsize=11)
                ax.set_ylabel(f'{feature_name} Value', fontsize=11)
                ax.legend(fontsize=9, loc='upper left')
                ax.grid(True, alpha=0.3)

                # Add shaded region for forecast horizon
                ax.axvspan(seq_len, seq_len + pred_len, alpha=0.1, color='blue')

                samples_plotted += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Horizon predictions plot saved to {save_path}')

    plt.show()


def plot_training_curves(train_losses, val_losses, test_loss=None, save_path=None):
    """
    Plot training, validation, and optionally test loss curves

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        test_loss: Final test loss (optional, shown as horizontal line)
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    epochs = range(1, len(train_losses) + 1)

    # Left plot: Training and validation curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=5)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=5)

    if test_loss is not None:
        ax1.axhline(y=test_loss, color='green', linestyle='--', linewidth=2, label=f'Test Loss ({test_loss:.6f})')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right plot: Loss gap (overfitting indicator)
    loss_gap = [val - train for train, val in zip(train_losses, val_losses)]
    ax2.plot(epochs, loss_gap, 'purple', linewidth=2.5, marker='d', markersize=5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(epochs, 0, loss_gap, alpha=0.3, color='purple')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss Gap (Val - Train)', fontsize=12)
    ax2.set_title('Generalization Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training curves saved to {save_path}')

    plt.show()


def plot_comprehensive_training_summary(train_losses, val_losses, test_results,
                                        dataset_name, seq_len, pred_len, save_path=None):
    """
    Create a comprehensive 4-panel training summary plot

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        test_results: Dictionary with test metrics (mse, mae, rmse)
        dataset_name: Name of the dataset
        seq_len: Input sequence length
        pred_len: Prediction horizon
        save_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    epochs = range(1, len(train_losses) + 1)

    # Panel 1: Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=2.5, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2.5, marker='s', markersize=4)
    ax1.axhline(y=test_results['mse'], color='green', linestyle='--', linewidth=2,
                label=f"Test ({test_results['mse']:.6f})")
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (MSE)', fontsize=11)
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Generalization gap
    ax2 = fig.add_subplot(gs[0, 1])
    loss_gap = [val - train for train, val in zip(train_losses, val_losses)]
    ax2.plot(epochs, loss_gap, 'purple', linewidth=2.5, marker='d', markersize=4)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(epochs, 0, loss_gap, alpha=0.3, color='purple')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Gap (Val - Train)', fontsize=11)
    ax2.set_title('Generalization Gap', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Test metrics comparison
    ax3 = fig.add_subplot(gs[1, 0])
    metrics = ['MSE', 'MAE', 'RMSE']
    values = [test_results['mse'], test_results['mae'], test_results['rmse']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.set_ylabel('Error Value', fontsize=11)
    ax3.set_title('Test Set Metrics', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Training summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary_text = f"""
    TRAINING SUMMARY
    {'='*40}

    Dataset: {dataset_name}
    Input Length: {seq_len}
    Prediction Horizon: {pred_len}

    {'='*40}
    FINAL RESULTS
    {'='*40}

    Final Train Loss:      {train_losses[-1]:.6f}
    Final Val Loss:        {val_losses[-1]:.6f}
    Best Val Loss:         {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses))+1})

    Test MSE:              {test_results['mse']:.6f}
    Test MAE:              {test_results['mae']:.6f}
    Test RMSE:             {test_results['rmse']:.6f}

    Total Epochs:          {len(train_losses)}
    Final Gap (Val-Train): {val_losses[-1] - train_losses[-1]:.6f}
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Main title
    fig.suptitle(f'{dataset_name} - Training Summary (seq_len={seq_len}, pred_len={pred_len})',
                fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Comprehensive training summary saved to {save_path}')

    plt.show()


def compare_metrics(results_dict, save_path=None):
    """
    Compare metrics across different experiments

    Args:
        results_dict: Dictionary of {experiment_name: results}
        save_path: Path to save figure (optional)
    """
    metrics = ['mse', 'mae', 'rmse', 'mape', 'mspe']
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 4))

    for idx, metric_name in enumerate(metrics):
        ax = axes[idx]
        exp_names = list(results_dict.keys())
        values = [results_dict[name]['test'][metric_name] for name in exp_names]

        ax.bar(exp_names, values, alpha=0.7)
        ax.set_title(metric_name.upper(), fontsize=12)
        ax.set_ylabel('Value', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Metrics comparison saved to {save_path}')

    plt.show()


if __name__ == '__main__':
    # Example usage
    print("Evaluation utilities for TimesNet")
    print("Import this module in your training script to use evaluation functions")
    print("\nAvailable functions:")
    print("  - evaluate_model(): Evaluate model on a dataset")
    print("  - evaluate_and_save_results(): Evaluate on train/val/test and save")
    print("  - visualize_predictions(): Plot predictions vs ground truth")
    print("  - plot_training_curves(): Plot loss curves")
    print("  - compare_metrics(): Compare multiple experiments")
