import torch
import pytest
from cliffeq.energy.zoo import NormEnergy, BilinearEnergy, GraphEnergy, HopfieldEnergy
from cliffordlayers.signature import CliffordSignature
from cliffeq.algebra.utils import geometric_product, reverse, scalar_part

@pytest.mark.parametrize("dim", [2, 3])
def test_bilinear_energy_correctness(dim):
    g = torch.ones(dim)
    B, Nin, Nh = 2, 3, 4
    energy_fn = BilinearEnergy(Nin, Nh, g)

    x = torch.randn(B, Nin, energy_fn.sig.n_blades)
    h = torch.randn(B, Nh, energy_fn.sig.n_blades)
    energy_fn.set_input(x)

    E = energy_fn(h)
    assert E.shape == (B,)

    # Manual calculation for one batch
    h0 = h[0] # (Nh, I)
    x0 = x[0] # (Nin, I)

    E_norm0 = 0.5 * torch.sum(h0**2)

    # Wx = W * x0
    # W: (Nh, Nin, I)
    # x0: (Nin, I)
    Wx0 = geometric_product(x0.unsqueeze(0), energy_fn.W, g).squeeze(0) # (Nh, I)

    # E_int = scalar(h̃ * Wx)
    h0_rev = reverse(h0, energy_fn.sig)
    res = geometric_product(h0_rev.unsqueeze(1), Wx0.unsqueeze(1), g) # (Nh, 1, I)
    E_int0 = scalar_part(res).sum()

    expected0 = E_norm0 - E_int0
    torch.testing.assert_close(E[0], expected0)

def test_norm_energy():
    energy_fn = NormEnergy()
    x = torch.randn(2, 3, 4)
    E = energy_fn(x)
    assert E.shape == (2,)
    torch.testing.assert_close(E[0], 0.5 * torch.sum(x[0]**2))

def test_hopfield_energy():
    g = torch.ones(2)
    M, N = 5, 10
    energy_fn = HopfieldEnergy(M, N, g)
    x = torch.randn(2, N, energy_fn.sig.n_blades)
    E = energy_fn(x)
    assert E.shape == (2,)
