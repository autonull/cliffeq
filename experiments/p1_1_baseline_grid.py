import torch
from torch import nn
import numpy as np
import time
from cliffeq.algebra.utils import embed_vector, scalar_part, reverse, geometric_product
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot, GeomProduct
from cliffeq.training.ep_engine import EPEngine
from cliffeq.benchmarks.metrics import equivariance_violation
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear

class EPModel(nn.Module):
    def __init__(self, energy_fn, dynamics_rule, n_free, n_clamped, beta, dt):
        super().__init__()
        self.energy_fn = energy_fn
        self.engine = EPEngine(energy_fn, dynamics_rule, n_free, n_clamped, beta, dt)
    def forward(self, x, h_init=None):
        if hasattr(self.energy_fn, 'set_input'): self.energy_fn.set_input(x)
        B = x.shape[0]
        comp = 1 if not hasattr(self.energy_fn, 'sig') else self.energy_fn.sig.n_blades
        if h_init is None:
            h_init = torch.zeros((B, self.energy_fn.hidden_dim, comp), device=x.device)
        h_free = self.engine.free_phase(h_init)
        return self.energy_fn.get_output(h_free)

def generate_data(n=2000):
    x = torch.randn(n, 2)
    y = (torch.sum(x**2, dim=1, keepdim=True) < 1.0).float()
    return x, y

def rotate_2d(x, angle):
    cost, sint = np.cos(angle), np.sin(angle)
    R = torch.tensor([[cost, -sint], [sint, cost]], device=x.device, dtype=x.dtype)
    return x @ R.t()

def run_p1_1():
    print("Phase 1.1: The Fundamental 2x2 Grid")
    n_samples = 2000
    x_train, y_train = generate_data(n_samples)
    x_test, y_test = generate_data(500)

    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    n_free, n_clamped, beta, dt = 100, 20, 1.0, 0.05
    loss_fn = nn.BCEWithLogitsLoss()

    # --- 1. Scalar BP ---
    model_sbp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1))
    opt_sbp = torch.optim.Adam(model_sbp.parameters(), lr=0.01)

    # --- 2. Clifford BP ---
    class CliffordBPModel(nn.Module):
        def __init__(self, g, nh):
            super().__init__()
            self.linear1 = CliffordLinear(g.tolist(), 1, nh, bias=True)
            self.linear2 = CliffordLinear(g.tolist(), nh, 1, bias=True)
        def forward(self, x):
            h = torch.relu(self.linear1(x))
            return self.linear2(h)[..., 0]
    model_cbp = CliffordBPModel(g, 32)
    opt_cbp = torch.optim.Adam(model_cbp.parameters(), lr=0.01)

    # --- 3. Scalar EP ---
    # Scalar BilinearEnergy with 1D signature (I=2 blades: 1, e1)
    # in_nodes = 1, but we have 2 scalar features.
    # Let's use 2 nodes each with 1D signature.
    energy_sep = BilinearEnergy(2, 128, torch.tensor([1.0]), use_spectral_norm=True)
    model_sep = EPModel(energy_sep, LinearDot(), n_free, n_clamped, beta, dt)
    opt_sep = torch.optim.Adam(model_sep.parameters(), lr=0.01)
    h_states_s = torch.zeros(n_samples, 128, 2)

    # --- 4. Clifford EP ---
    energy_cep = BilinearEnergy(1, 64, g, use_spectral_norm=True)
    model_cep = EPModel(energy_cep, LinearDot(), n_free, n_clamped, beta, dt)
    opt_cep = torch.optim.Adam(model_cep.parameters(), lr=0.01)
    h_states_c = torch.zeros(n_samples, 64, 4)

    print("Training...")
    sig1d = CliffordSignature(torch.tensor([1.0]))
    for epoch in range(30):
        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, 64):
            idx = indices[i:i+64]
            if len(idx) < 64: continue
            xb, yb = x_train[idx], y_train[idx]
            xb_mv = embed_vector(xb.unsqueeze(1), sig)

            # Scalar BP
            opt_sbp.zero_grad(); loss_fn(model_sbp(xb), yb).backward(); opt_sbp.step()
            # Clifford BP
            opt_cbp.zero_grad(); loss_fn(model_cbp(xb_mv), yb).backward(); opt_cbp.step()

            # Scalar EP
            xb_s_mv = embed_vector(xb.unsqueeze(-1), sig1d)
            model_sep.energy_fn.set_input(xb_s_mv)
            h_fs = model_sep.engine.free_phase(h_states_s[idx])
            h_states_s[idx] = h_fs.detach()
            h_cs = model_sep.engine.clamped_phase(h_fs, yb, lambda h, t: loss_fn(model_sep.energy_fn.get_output(h), t))
            model_sep.engine.parameter_update(h_fs, h_cs)
            opt_sep.step(); opt_sep.zero_grad()

            # Clifford EP
            model_cep.energy_fn.set_input(xb_mv)
            h_fc = model_cep.engine.free_phase(h_states_c[idx])
            h_states_c[idx] = h_fc.detach()
            h_cc = model_cep.engine.clamped_phase(h_fc, yb, lambda h, t: loss_fn(model_cep.energy_fn.get_output(h), t))
            model_cep.engine.parameter_update(h_fc, h_cc)
            opt_cep.step(); opt_cep.zero_grad()

        if epoch % 5 == 0:
            with torch.no_grad():
                xt_mv = embed_vector(x_test.unsqueeze(1), sig)
                xt_s_mv = embed_vector(x_test.unsqueeze(-1), sig1d)
                acc_sbp = ((model_sbp(x_test) > 0) == y_test).float().mean().item()
                acc_cbp = ((model_cbp(xt_mv) > 0) == y_test).float().mean().item()
                acc_sep = ((model_sep(xt_s_mv) > 0) == y_test).float().mean().item()
                acc_cep = ((model_cep(xt_mv) > 0) == y_test).float().mean().item()
                print(f"Epoch {epoch}: S-BP:{acc_sbp:.2f} C-BP:{acc_cbp:.2f} S-EP:{acc_sep:.2f} C-EP:{acc_cep:.2f}")

    def transform_x(x):
        res = x.clone()
        if x.ndim == 2: return rotate_2d(x, np.pi/2)
        if x.shape[-1] == 4:
            res[..., 0, 1], res[..., 0, 2] = -x[..., 0, 2], x[..., 0, 1]
        elif x.shape[-1] == 2:
            res[..., 0, 1] = -x[..., 1, 1]
            res[..., 1, 1] = x[..., 0, 1]
        return res

    xt_mv = embed_vector(x_test.unsqueeze(1), sig)
    xt_s_mv = embed_vector(x_test.unsqueeze(-1), sig1d)
    v_sbp = equivariance_violation(model_sbp, x_test, lambda x: rotate_2d(x, np.pi/2), lambda y: y)
    v_cbp = equivariance_violation(model_cbp, xt_mv, transform_x, lambda y: y)
    v_sep = equivariance_violation(model_sep, xt_s_mv, transform_x, lambda y: y)
    v_cep = equivariance_violation(model_cep, xt_mv, transform_x, lambda y: y)

    print(f"\nEquivariance Violation (lower is better):")
    print(f"Scalar BP:   {v_sbp:.4f}")
    print(f"Clifford BP: {v_cbp:.4f}")
    print(f"Scalar EP:   {v_sep:.4f}")
    print(f"Clifford EP: {v_cep:.4f}")

if __name__ == "__main__":
    run_p1_1()
