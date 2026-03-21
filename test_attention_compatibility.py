import torch
import torch.nn as nn
from cliffeq.attention.geometric import CliffordAttention

def test_drop_in_compatibility():
    print("Testing CliffordAttention drop-in compatibility...")

    batch_size = 2
    seq_len = 8
    n_heads = 4
    clifford_dim = 2
    n_blades = 4 # Cl(2,0)
    d_model = n_heads * clifford_dim * n_blades

    sig_g = torch.tensor([1.0, 1.0])

    # 1. Test standard forward
    model = CliffordAttention(n_heads, clifford_dim, sig_g)
    x = torch.randn(batch_size, seq_len, d_model)

    out, weights = model(x, x, x)

    print(f"Output shape: {out.shape}")
    assert out.shape == (batch_size, seq_len, d_model)
    print(f"Weights shape: {weights.shape}")
    assert weights.shape == (batch_size, seq_len, seq_len)

    # 2. Test with attn_mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9
    out_masked, weights_masked = model(x, x, x, attn_mask=mask)

    assert out_masked.shape == (batch_size, seq_len, d_model)
    # Check if upper triangle of weights is zero (or very small)
    print(f"Max upper triangle weight: {weights_masked[:, 0, 1:].max().item()}")
    assert weights_masked[:, 0, 1:].max() < 1e-6

    print("✓ CliffordAttention drop-in compatibility verified")

if __name__ == "__main__":
    test_drop_in_compatibility()
