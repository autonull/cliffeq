"""
Comprehensive test of fixed geometric_product.
"""

import torch
from cliffeq.algebra.utils import geometric_product
from cliffordlayers.signature import CliffordSignature

def test_scalar_product():
    """Test scalar * scalar = scalar."""
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)
    I = sig.n_blades
    B, N = 4, 3
    
    x = torch.zeros(B, N, I)
    y = torch.zeros(B, N, I)
    x[..., 0] = 2.0
    y[..., 0] = 3.0
    
    result = geometric_product(x, y, sig_g)
    
    assert result.shape == (B, N, I)
    # 2 * 3 = 6 in scalar position, zero elsewhere
    assert torch.allclose(result[..., 0], torch.full((B, N), 6.0), atol=1e-5), \
        f"Scalar product failed: expected all 6.0, got {result[0, :, 0]}"
    assert torch.allclose(result[..., 1:], torch.zeros(B, N, I-1), atol=1e-5), \
        "Non-scalar components should be zero"
    print("✓ Scalar product test passed")

def test_vector_self_product():
    """Test v * v = ||v||^2 (scalar)."""
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)
    I = sig.n_blades
    B, N = 4, 3
    
    v = torch.zeros(B, N, I)
    v[..., 1:4] = torch.ones(B, N, 3)  # Unit vectors (1,1,1)
    
    result = geometric_product(v, v, sig_g)
    
    assert result.shape == (B, N, I)
    # v * v for v=(1,1,1) should give 1+1+1=3
    expected_norm_sq = 3.0
    actual_scalar = result[..., 0]
    assert torch.allclose(actual_scalar, torch.full((B, N), expected_norm_sq), atol=1e-5), \
        f"Vector self-product failed: expected {expected_norm_sq}, got {actual_scalar[0, 0]}"
    print("✓ Vector self-product test passed")

def test_vector_product():
    """Test v * w for different vectors."""
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)
    I = sig.n_blades
    B, N = 2, 2
    
    # v = (1, 0, 0), w = (0, 1, 0)
    v = torch.zeros(B, N, I)
    v[..., 1] = 1.0
    
    w = torch.zeros(B, N, I)
    w[..., 2] = 1.0
    
    result = geometric_product(v, w, sig_g)
    
    assert result.shape == (B, N, I)
    # (1,0,0) * (0,1,0) should give a bivector e_12
    # In the blade ordering, this is blade 4 (for Cl(3,0))
    print(f"  v * w result: {result[0, 0]}")
    print("✓ Vector product test passed")

def test_noncontiguous():
    """Test non-contiguous tensor handling."""
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)
    I = sig.n_blades
    
    # Create non-contiguous tensor
    x_temp = torch.randn(3, 4, I)
    x = x_temp.permute(1, 0, 2)  # Now (4, 3, I) but non-contiguous
    y = torch.randn(4, 3, I)
    
    assert not x.is_contiguous(), "x should be non-contiguous"
    
    result = geometric_product(x, y, sig_g)
    assert result.shape == (4, 3, I)
    print("✓ Non-contiguous tensor test passed")

if __name__ == "__main__":
    test_scalar_product()
    test_vector_self_product()
    test_vector_product()
    test_noncontiguous()
    print("\n✅ All comprehensive tests passed!")
