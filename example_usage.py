"""
Example usage of enhanced visualization functions
Add these cells to your train.ipynb after training
"""

# ============================================
# After training a model, add these visualizations
# ============================================

from src.evaluate import (
    visualize_horizon_predictions,
    plot_comprehensive_training_summary,
    evaluate_and_save_results
)

# 1. Visualize horizon predictions (input context + forecast)
# Shows 3 samples from train/val/test with the full forecasting context
print("Creating horizon prediction visualizations...")
visualize_horizon_predictions(
    model=trainer.model,
    train_loader=trainer.train_loader,
    val_loader=trainer.val_loader,
    test_loader=trainer.test_loader,
    device=trainer.device,
    scaler=trainer.train_dataset.scaler,
    seq_len=config['seq_len'],
    pred_len=config['pred_len'],
    num_samples=3,  # Show 3 samples per split
    feature_idx=0,  # First feature (usually 'OT')
    feature_name='OT',
    save_path=f'{RESULTS_DIR}/{dataset_name}_{SEQ_LEN}_{pred_len}_horizons.png'
)

# 2. Comprehensive training summary (4-panel plot)
# Shows loss curves, generalization gap, metrics, and text summary
print("Creating comprehensive training summary...")
plot_comprehensive_training_summary(
    train_losses=train_losses,
    val_losses=val_losses,
    test_results=test_results,
    dataset_name=dataset_name,
    seq_len=config['seq_len'],
    pred_len=config['pred_len'],
    save_path=f'{RESULTS_DIR}/{dataset_name}_{SEQ_LEN}_{pred_len}_summary.png'
)

# 3. Evaluate on all splits with detailed metrics
print("Evaluating on all splits...")
all_split_results = evaluate_and_save_results(
    model=trainer.model,
    train_loader=trainer.train_loader,
    val_loader=trainer.val_loader,
    test_loader=trainer.test_loader,
    device=trainer.device,
    scaler=trainer.train_dataset.scaler,
    save_path=f'{RESULTS_DIR}/{dataset_name}_{SEQ_LEN}_{pred_len}_all_metrics.txt'
)

# ============================================
# Example: Update train_single_model function
# ============================================

def train_single_model_enhanced(dataset_name, pred_len):
    """
    Enhanced training function with comprehensive visualizations
    """
    print("\n" + "="*70)
    print(f"Training: {dataset_name} | seq_len={SEQ_LEN} → pred_len={pred_len}")
    print("="*70)

    # Create config
    config = {
        'root_path': ROOT_PATH,
        'data_path': f'{dataset_name}.csv',
        'seq_len': SEQ_LEN,
        'pred_len': pred_len,
        'checkpoints': f'{CHECKPOINT_BASE}/{dataset_name}_{SEQ_LEN}_{pred_len}',
        **TRAIN_CONFIG,
        **MODEL_CONFIGS[dataset_name]
    }

    os.makedirs(config['checkpoints'], exist_ok=True)

    # Train
    trainer = TimesNetTrainer(config)
    train_losses, val_losses = trainer.train()

    # Test
    test_results = trainer.test()

    # ========== NEW VISUALIZATIONS ==========

    # 1. Horizon predictions (input + forecast)
    print("\n📊 Creating horizon predictions...")
    visualize_horizon_predictions(
        model=trainer.model,
        train_loader=trainer.train_loader,
        val_loader=trainer.val_loader,
        test_loader=trainer.test_loader,
        device=trainer.device,
        scaler=trainer.train_dataset.scaler,
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        num_samples=3,
        feature_name='OT',
        save_path=f'{RESULTS_DIR}/{dataset_name}_{SEQ_LEN}_{pred_len}_horizons.png'
    )

    # 2. Comprehensive summary
    print("📊 Creating comprehensive summary...")
    plot_comprehensive_training_summary(
        train_losses=train_losses,
        val_losses=val_losses,
        test_results=test_results,
        dataset_name=dataset_name,
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        save_path=f'{RESULTS_DIR}/{dataset_name}_{SEQ_LEN}_{pred_len}_summary.png'
    )

    # 3. All splits evaluation
    print("📊 Evaluating all splits...")
    all_split_results = evaluate_and_save_results(
        model=trainer.model,
        train_loader=trainer.train_loader,
        val_loader=trainer.val_loader,
        test_loader=trainer.test_loader,
        device=trainer.device,
        scaler=trainer.train_dataset.scaler,
        save_path=f'{RESULTS_DIR}/{dataset_name}_{SEQ_LEN}_{pred_len}_all_metrics.txt'
    )

    # Prepare results
    results = {
        'dataset': dataset_name,
        'seq_len': SEQ_LEN,
        'pred_len': pred_len,
        'd_model': config['d_model'],
        'd_ff': config['d_ff'],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_mse': test_results['mse'],
        'test_mae': test_results['mae'],
        'test_rmse': test_results['rmse'],
        'final_epoch': len(train_losses),
        # Add all split results
        'train_mse': all_split_results['train']['mse'],
        'val_mse': all_split_results['val']['mse'],
    }

    print(f"\n✓ Completed: {dataset_name}_{SEQ_LEN}_{pred_len}")
    print(f"  - Train MSE: {all_split_results['train']['mse']:.6f}")
    print(f"  - Val MSE:   {all_split_results['val']['mse']:.6f}")
    print(f"  - Test MSE:  {test_results['mse']:.6f}")

    return results
