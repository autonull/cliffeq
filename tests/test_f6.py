import torch
from torch import nn
from cliffeq.attention.geometric import CliffordAttention

def test_f6_attn():
    g = torch.tensor([1.0, 1.0])
    attn_layer = CliffordAttention(n_heads=2, clifford_dim=1, sig_g=g)
    x = torch.randn(1, 10, 8)
    out = attn_layer(x)
    assert out.shape == (1, 10, 8)
    print("test_f6_attn passed")

if __name__ == "__main__":
    test_f6_attn()
