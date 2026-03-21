import torch
import pytest
from cliffeq.energy.zoo import NormEnergy, BilinearEnergy, GraphEnergy, HopfieldEnergy, GradeWeightedEnergy
from cliffordlayers.signature import CliffordSignature
from cliffeq.algebra.utils import geometric_product, reverse, scalar_part, get_blade_signs

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
    Wx0 = energy_fn.W_layer(x0.unsqueeze(0)).squeeze(0) # (Nh, I)

    # scalar(rev(rho(h)) * Wx)
    # rho(h) = clamp(h, 0, 1)
    rho_h0 = torch.clamp(h0, 0, 1)
    signs = get_blade_signs(energy_fn.sig, h0.device)
    E_int0 = torch.sum(rho_h0 * Wx0 * signs)

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

def test_graph_energy():
    g = torch.ones(2)
    N = 5
    energy_fn = GraphEnergy(N, g)
    x = torch.randn(2, N, energy_fn.sig.n_blades)
    E = energy_fn(x)
    assert E.shape == (2,)

def test_grade_weighted_energy():
    g = torch.ones(2)
    energy_fn = GradeWeightedEnergy(g)
    x = torch.randn(2, 5, 4)
    E = energy_fn(x)
    assert E.shape == (2,)
