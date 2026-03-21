import torch
import pytest
from cliffeq.energy.zoo import NormEnergy, BilinearEnergy
from cliffeq.dynamics.rules import LinearDot, GeomProduct, RotorOnly, WedgeUpdate, Riemannian, GradeSplit
from cliffordlayers.signature import CliffordSignature

def test_dynamics_convergence():
    dim = 2
    g = torch.ones(dim)
    energy_fn = NormEnergy()
    x = torch.randn(1, 1, 4)
    alpha = 0.1

    # Linear rules should converge with small alpha
    for rule in [LinearDot(), Riemannian()]:
        curr_x = x.clone()
        prev_e = energy_fn(curr_x)
        for _ in range(10):
            curr_x = rule.step(curr_x, energy_fn, alpha)
            curr_e = energy_fn(curr_x)
            assert curr_e <= prev_e + 1e-6, f"Rule {rule.__class__.__name__} increased energy"
            prev_e = curr_e

    # Geometric rules are non-linear updates, may not monotonically decrease NormEnergy
    # but should still stay stable.
    for rule in [GeomProduct(g, normalize=True), RotorOnly(g)]:
        curr_x = x.clone()
        for _ in range(10):
            curr_x = rule.step(curr_x, energy_fn, alpha * 0.01)
        curr_e = energy_fn(curr_x)
        assert not torch.isnan(curr_e).any(), f"Rule {rule.__class__.__name__} produced NaNs"

def test_wedge_update():
    dim = 2
    g = torch.ones(dim)
    sig = CliffordSignature(g)
    energy_fn = NormEnergy()
    x = torch.randn(1, 1, 4)
    rule = WedgeUpdate(g)

    # Just check it doesn't crash and returns correct shape
    out = rule.step(x, energy_fn, 0.1)
    assert out.shape == x.shape

def test_grade_split():
    energy_fn = NormEnergy()
    x = torch.randn(1, 1, 4)
    rule = GradeSplit({0: 0.1, 1: 0.2, 2: 0.3})
    out = rule.step(x, energy_fn, 0.1)
    assert out.shape == x.shape
