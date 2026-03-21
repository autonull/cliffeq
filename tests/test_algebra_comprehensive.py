import torch
import pytest
from cliffeq.algebra.utils import geometric_product, clifford_norm_sq, scalar_part, reverse, embed_scalar, embed_vector
from cliffordlayers.signature import CliffordSignature

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_geometric_product_elementwise(dim):
    g = torch.ones(dim)
    sig = CliffordSignature(g)
    I = sig.n_blades
    B, N = 2, 3

    x = torch.randn(B, N, I)
    y = torch.randn(B, N, I)

    # Current implementation uses a loop
    out = geometric_product(x, y, g)

    assert out.shape == (B, N, I)

    # Verify a single element
    from cliffordlayers.cliffordkernels import get_1d_clifford_kernel, get_2d_clifford_kernel, get_3d_clifford_kernel
    kernel_fn = {1: get_1d_clifford_kernel, 2: get_2d_clifford_kernel, 3: get_3d_clifford_kernel}[dim]

    x00 = x[0, 0]
    y00 = y[0, 0]
    res = kernel_fn(y00.view(I, 1, 1), g)
    kernel = res[1] if isinstance(res, tuple) else res
    expected00 = torch.matmul(kernel, x00)

    torch.testing.assert_close(out[0, 0], expected00)

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_geometric_product_weight_based(dim):
    g = torch.ones(dim)
    sig = CliffordSignature(g)
    I = sig.n_blades
    B, Nin, Nout = 2, 4, 5

    x = torch.randn(B, Nin, I)
    W = torch.randn(Nout, Nin, I)

    out = geometric_product(x, W, g)

    assert out.shape == (B, Nout, I)

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_clifford_norm_sq(dim):
    g = torch.ones(dim)
    sig = CliffordSignature(g)
    I = sig.n_blades
    x = torch.zeros(1, I)
    x[0, 0] = 1.0 # scalar 1

    n2 = clifford_norm_sq(x, sig)
    torch.testing.assert_close(n2, torch.tensor([1.0]))

    if dim >= 1:
        x = torch.zeros(1, I)
        x[0, 1] = 1.0 # e1
        n2 = clifford_norm_sq(x, sig)
        # reverse(e1)*e1 = e1*e1 = g1 = 1
        torch.testing.assert_close(n2, torch.tensor([1.0]))

    if dim >= 2:
        x = torch.zeros(1, I)
        x[0, 3] = 1.0 # e12
        n2 = clifford_norm_sq(x, sig)
        # reverse(e12)*e12 = -e12*e12 = -(-1) = 1 (for g=1,1)
        torch.testing.assert_close(n2, torch.tensor([1.0]))

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_scalar_part_of_product(dim):
    g = torch.ones(dim)
    sig = CliffordSignature(g)
    I = sig.n_blades
    x = torch.randn(1, 1, I)
    y = torch.randn(1, 1, I)

    # scalar(reverse(x) * y)
    from cliffeq.algebra.utils import get_blade_signs
    signs = get_blade_signs(sig, x.device)
    x_rev = reverse(x, sig)

    # Method 1: full GP then scalar part
    prod = geometric_product(x_rev, y, g)
    s1 = scalar_part(prod)

    # Method 2: dot product with signs
    s2 = torch.sum(x * y * signs, dim=-1)

    torch.testing.assert_close(s1, s2)
