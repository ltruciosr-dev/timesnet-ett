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


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training curves saved to {save_path}')

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
