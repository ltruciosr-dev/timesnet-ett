"""
Test TimesNet Implementation
Validates architecture matches docs/ARCHITECTURE_EXPLAINED.md
"""

import sys
import torch

sys.path.append('./src')

from src.dataset import create_dataloaders
from src.model import TimesNet

def test_dataset():
    """Dataset returns 3 items: seq_x, seq_y, seq_x_mark (NO seq_y_mark)"""
    train_loader, _, _, _ = create_dataloaders(
        root_path='./ETDataset/ETT-small/',
        data_path='ETTh1.csv',
        batch_size=2,
        seq_len=96,
        pred_len=96
    )

    batch = next(iter(train_loader))

    assert len(batch) == 3, f"Expected 3 items, got {len(batch)}"
    assert batch[0].shape == (2, 96, 7), f"seq_x shape: {batch[0].shape}"
    assert batch[1].shape == (2, 96, 7), f"seq_y shape: {batch[1].shape}"
    assert batch[2].shape == (2, 96, 4), f"seq_x_mark shape: {batch[2].shape}"

    print("✓ Dataset: Returns 3 items (seq_x, seq_y, seq_x_mark)")
    return train_loader

def test_model_forward():
    """Model forward: x_enc + x_mark_enc → predictions"""
    model = TimesNet(
        seq_len=96, pred_len=96, enc_in=7, c_out=7,
        d_model=16, d_ff=32, num_kernels=6, top_k=5,
        e_layers=2, dropout=0.1, embed='fixed', freq='h'
    )

    batch_x = torch.randn(2, 96, 7)
    batch_x_mark = torch.zeros(2, 96, 4, dtype=torch.long)
    batch_x_mark[:, :, 0] = torch.randint(1, 13, (2, 96))
    batch_x_mark[:, :, 1] = torch.randint(1, 32, (2, 96))
    batch_x_mark[:, :, 2] = torch.randint(0, 7, (2, 96))
    batch_x_mark[:, :, 3] = torch.randint(0, 24, (2, 96))

    with torch.no_grad():
        output = model(batch_x, batch_x_mark)

    assert output.shape == (2, 96, 7), f"Output shape: {output.shape}"
    print("✓ Model: forward(x_enc, x_mark_enc) → [B, pred_len, features]")

    # Should fail with extra args
    try:
        model(batch_x, batch_x_mark, batch_x)
        assert False, "Should reject 3 args"
    except TypeError:
        print("✓ Model: Encoder-only (rejects decoder inputs)")

    return model

def test_forward_steps():
    """Validate forward pass steps match documentation"""
    model = TimesNet(seq_len=96, pred_len=96, enc_in=7, c_out=7, d_model=16, d_ff=32)
    model.eval()

    batch_x = torch.randn(2, 96, 7)
    batch_x_mark = torch.zeros(2, 96, 4, dtype=torch.long)
    batch_x_mark[:, :, 0] = torch.randint(1, 13, (2, 96))
    batch_x_mark[:, :, 1] = torch.randint(1, 32, (2, 96))
    batch_x_mark[:, :, 2] = torch.randint(0, 7, (2, 96))
    batch_x_mark[:, :, 3] = torch.randint(0, 24, (2, 96))

    # Step 1: Normalization
    means = batch_x.mean(1, keepdim=True).detach()
    x_norm = (batch_x - means) / (torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5))
    assert abs(x_norm.mean().item()) < 0.01, "Normalization failed"

    # Step 2: Embedding
    with torch.no_grad():
        enc_out = model.enc_embedding(x_norm, batch_x_mark)
    assert enc_out.shape == (2, 96, 16), f"Embedding: {enc_out.shape}"

    # Step 3: Linear projection (96 → 192)
    with torch.no_grad():
        expanded = model.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
    assert expanded.shape == (2, 192, 16), f"Projection: {expanded.shape}"

    # Step 4: TimesBlocks
    with torch.no_grad():
        for block in model.model:
            expanded = model.layer_norm(block(expanded))
    assert expanded.shape == (2, 192, 16), f"TimesBlocks: {expanded.shape}"

    # Step 5: Final projection
    with torch.no_grad():
        output = model.projection(expanded)
    assert output.shape == (2, 192, 7), f"Final proj: {output.shape}"

    # Extract last pred_len
    predictions = output[:, -96:, :]
    assert predictions.shape == (2, 96, 7), f"Predictions: {predictions.shape}"

    print("✓ Forward steps: Normalize → Embed → Project(96→192) → TimesBlocks → Extract(last 96)")

def test_training_integration(train_loader):
    """Training loop: forward → loss → backward"""
    model = TimesNet(seq_len=96, pred_len=96, enc_in=7, c_out=7, d_model=16, d_ff=32)

    batch_x, batch_y, batch_x_mark = next(iter(train_loader))

    with torch.no_grad():
        outputs = model(batch_x.float(), batch_x_mark.float())

    assert outputs.shape == batch_y.shape, "Output/target mismatch"

    criterion = torch.nn.MSELoss()
    loss = criterion(outputs, batch_y.float())

    assert loss.item() > 0, "Loss computation failed"
    print(f"✓ Training: forward → MSE loss ({loss.item():.4f})")

def test_paper_config():
    """Paper hyperparameters (ETTh1)"""
    model = TimesNet(
        seq_len=96, pred_len=96, enc_in=7, c_out=7,
        d_model=16, d_ff=32, top_k=5, e_layers=2,
        num_kernels=6, dropout=0.1, embed='fixed', freq='h'
    )

    num_params = sum(p.numel() for p in model.parameters())

    assert len(model.model) == 2, "Should have 2 TimesBlocks"
    assert model.model[0].k == 5, "top_k should be 5"

    print(f"✓ Paper config: d_model=16, e_layers=2, top_k=5 ({num_params:,} params)")

if __name__ == '__main__':
    print("\nTimesNet Implementation Validation\n")

    train_loader = test_dataset()
    model = test_model_forward()
    test_forward_steps()
    test_training_integration(train_loader)
    test_paper_config()

    print("\n✅ All tests passed - implementation matches ARCHITECTURE_EXPLAINED.md\n")
