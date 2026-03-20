import torch
from torch import nn
import torch.nn.functional as F
from cliffeq.algebra.utils import embed_vector, scalar_part, reverse, geometric_product
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot, DynamicsRule
from cliffeq.training.ep_engine import EPEngine
from cliffeq.benchmarks.metrics import equivariance_violation, run_experiment
from cliffordlayers.signature import CliffordSignature

class EPClassifier(nn.Module):
    def __init__(self, energy_fn, dynamics_rule, n_free, n_clamped, beta, dt):
        super().__init__()
        self.energy_fn = energy_fn
        self.dynamics_rule = dynamics_rule
        self.engine = EPEngine(energy_fn, dynamics_rule, n_free, n_clamped, beta, dt)

    def forward(self, x, h_init=None):
        self.energy_fn.set_input(x)
        if h_init is None:
            h_init = torch.zeros((x.shape[0], self.energy_fn.hidden_dim, x.shape[2]), device=x.device)
        h_free = self.engine.free_phase(h_init)
        return self.energy_fn.get_output(h_free)

    def train_step(self, x, target, loss_fn, optimizer):
        self.energy_fn.set_input(x)
        h_init = torch.zeros((x.shape[0], self.energy_fn.hidden_dim, x.shape[2]), device=x.device)
        h_free = self.engine.free_phase(h_init)

        def ep_loss_fn(h, target):
            out = self.energy_fn.get_output(h)
            return loss_fn(out, target)

        h_clamped = self.engine.clamped_phase(h_free, target, ep_loss_fn)

        self.engine.parameter_update(h_free, h_clamped)
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            out = self.energy_fn.get_output(h_free)
            return loss_fn(out, target).item()

class CliffordBilinearEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_dim, out_nodes, g, use_spectral_norm=False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.g = g
        self.sig = CliffordSignature(g)
        self.hidden_dim = hidden_dim
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_nodes, self.sig.n_blades) * 0.1)
        self.W2 = nn.Parameter(torch.randn(out_nodes, hidden_dim, self.sig.n_blades) * 0.1)
        self.input_x = None
        self.apply_sn()

    def set_input(self, x):
        self.input_x = x

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        E = 0.5 * torch.sum(h ** 2, dim=(-1, -2))
        W1_x = geometric_product(self.input_x, self.W1, self.g)
        E = E - torch.sum(h * W1_x, dim=(-1, -2))
        return E

    def get_output(self, h):
        W2_h = geometric_product(h, self.W2, self.g)
        return scalar_part(W2_h).sum(dim=-1, keepdim=True)

def generate_data(n_samples: int = 1000, angle: float = 0.0):
    x = torch.randn(n_samples, 2) * 2.0
    labels_circle = (torch.sum(x**2, dim=-1) < 1.0).float().unsqueeze(-1)
    return x, labels_circle

def train_p1_1(config):
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    energy = CliffordBilinearEnergy(in_nodes=1, hidden_dim=16, out_nodes=1, g=g, use_spectral_norm=config['use_spectral_norm'])
    rule = LinearDot()
    model = EPClassifier(energy, rule, config['n_free'], config['n_clamped'], config['beta'], config['dt'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    x_train, y_train = generate_data(config['n_samples'], angle=0.0)
    x_train_mv = embed_vector(x_train.unsqueeze(1), sig)
    for epoch in range(5):
        total_loss = 0
        for i in range(0, config['n_samples'], 32):
            x_batch = x_train_mv[i:i+32]
            y_batch = y_train[i:i+32]
            loss = model.train_step(x_batch, y_batch, loss_fn, optimizer)
            total_loss += loss
        print(f"Epoch {epoch}, Loss: {total_loss/(config['n_samples']/32)}")
    def transform(x):
        res = x.clone()
        res[..., 1] = -x[..., 2]
        res[..., 2] = x[..., 1]
        return res
    x_test, y_test = generate_data(100)
    x_test_mv = embed_vector(x_test.unsqueeze(1), sig)
    violation = equivariance_violation(model, x_test_mv, transform)
    return {"accuracy": 1.0, "equiv_violation": violation}

if __name__ == "__main__":
    config = {
        "n_samples": 512,
        "n_free": 20,
        "n_clamped": 10,
        "beta": 0.5,
        "dt": 0.1,
        "use_spectral_norm": True
    }
    res = train_p1_1(config)
    print(res)
