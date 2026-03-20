import torch
from torch import nn
import torch.nn.functional as F
from cliffeq.algebra.utils import embed_vector, scalar_part, reverse, geometric_product
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot
from cliffeq.training.ep_engine import EPEngine
from cliffeq.benchmarks.metrics import equivariance_violation
from cliffordlayers.signature import CliffordSignature

class ScalarBilinearEnergy(EnergyFunction):
    def __init__(self, in_dim, hidden_dim, out_dim, use_spectral_norm=False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.input_x = None
        self.apply_sn()
    def set_input(self, x):
        self.input_x = x
    def forward(self, h):
        w1x = self.w1(self.input_x)
        if h.dim() == 3: # (B, N, 1)
            h = h.squeeze(-1)
        return 0.5 * torch.sum(h ** 2, dim=-1) - torch.sum(h * w1x, dim=-1)
    def get_output(self, h):
        if h.dim() == 3:
            h = h.squeeze(-1)
        return self.w2(h)

class CliffordBilinearEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_dim, out_nodes, g, use_spectral_norm=False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.g = g
        self.sig = CliffordSignature(g)
        self.hidden_dim = hidden_dim
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_nodes, self.sig.n_blades) * 0.1)
        self.W2 = nn.Parameter(torch.randn(out_nodes, hidden_dim, self.sig.n_blades) * 0.1)
        self.input_x = None
        self.apply_sn()
    def set_input(self, x):
        self.input_x = x
    def forward(self, h):
        E = 0.5 * torch.sum(h ** 2, dim=(-1, -2))
        W1_x = geometric_product(self.input_x, self.W1, self.g)
        E = E - torch.sum(h * W1_x, dim=(-1, -2))
        return E
    def get_output(self, h):
        W2_h = geometric_product(h, self.W2, self.g)
        return scalar_part(W2_h).sum(dim=-1, keepdim=True)

class EPModel(nn.Module):
    def __init__(self, energy_fn, dynamics_rule, n_free, n_clamped, beta, dt):
        super().__init__()
        self.energy_fn = energy_fn
        self.engine = EPEngine(energy_fn, dynamics_rule, n_free, n_clamped, beta, dt)
    def forward(self, x):
        self.energy_fn.set_input(x)
        # Components dimension: 1 for scalar, sig.n_blades for clifford
        comp = 1 if not hasattr(self.energy_fn, 'sig') else self.energy_fn.sig.n_blades
        h_init = torch.zeros((x.shape[0], self.energy_fn.hidden_dim, comp), device=x.device)
        h_free = self.engine.free_phase(h_init)
        return self.energy_fn.get_output(h_free)
    def train_step(self, x, target, optimizer):
        self.energy_fn.set_input(x)
        comp = 1 if not hasattr(self.energy_fn, 'sig') else self.energy_fn.sig.n_blades
        h_init = torch.zeros((x.shape[0], self.energy_fn.hidden_dim, comp), device=x.device)
        h_free = self.engine.free_phase(h_init)
        def ep_loss_fn(h, target):
            out = self.energy_fn.get_output(h)
            return 0.5 * torch.sum((out - target) ** 2)
        h_clamped = self.engine.clamped_phase(h_free, target, ep_loss_fn)
        self.engine.parameter_update(h_free, h_clamped)
        optimizer.step()
        optimizer.zero_grad()

def generate_data(n=512):
    x = torch.randn(n, 2)
    y = (torch.norm(x, dim=1, keepdim=True) < 1.0).float()
    return x, y

def run_p1_1():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    n_free, n_clamped, beta, dt = 20, 10, 0.5, 0.1

    x_train, y_train = generate_data(512)
    x_test, y_test = generate_data(128)

    # 1. Scalar EP
    energy_scalar = ScalarBilinearEnergy(2, 16, 1)
    model_scalar = EPModel(energy_scalar, LinearDot(), n_free, n_clamped, beta, dt)
    opt_scalar = torch.optim.Adam(model_scalar.parameters(), lr=0.01)

    # 2. Clifford EP
    energy_cliff = CliffordBilinearEnergy(1, 16, 1, g)
    model_cliff = EPModel(energy_cliff, LinearDot(), n_free, n_clamped, beta, dt)
    opt_cliff = torch.optim.Adam(model_cliff.parameters(), lr=0.01)

    print("Training models...")
    for epoch in range(5):
        for i in range(0, 512, 32):
            model_scalar.train_step(x_train[i:i+32], y_train[i:i+32], opt_scalar)
            model_cliff.train_step(embed_vector(x_train[i:i+32].unsqueeze(1), sig), y_train[i:i+32], opt_cliff)

    def transform(x):
        if x.shape[-1] == 2: # scalar input
            return torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        # multivector input [1, e1, e2, e12]
        res = x.clone()
        res[..., 1] = -x[..., 2]
        res[..., 2] = x[..., 1]
        return res

    v_scalar = equivariance_violation(model_scalar, x_test, transform)
    v_cliff = equivariance_violation(model_cliff, embed_vector(x_test.unsqueeze(1), sig), transform)

    print(f"Scalar EP Equiv Violation: {v_scalar:.4f}")
    print(f"Clifford EP Equiv Violation: {v_cliff:.4f}")

if __name__ == "__main__":
    run_p1_1()
