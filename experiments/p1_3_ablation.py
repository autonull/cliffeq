import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product, grade_project
from cliffeq.training.ep_engine import EPEngine
from cliffordlayers.signature import CliffordSignature
import time

class CliffordBilinearEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_dim, out_nodes, g, allowed_grades):
        super().__init__()
        self.g = g
        self.sig = CliffordSignature(g)
        self.allowed_grades = allowed_grades
        self.hidden_dim = hidden_dim
        # Parameter matched setup: total parameters should be similar
        # Here we fix architecture and just zero out grades.
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_nodes, self.sig.n_blades) * 0.1)
        self.W2 = nn.Parameter(torch.randn(out_nodes, hidden_dim, self.sig.n_blades) * 0.1)
        self.input_x = None
    def set_input(self, x):
        self.input_x = x
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = grade_project(h, self.allowed_grades, self.sig)
        E = 0.5 * torch.sum(h ** 2, dim=(-1, -2))
        W1_x = geometric_product(self.input_x, self.W1, self.g)
        E = E - torch.sum(h * W1_x, dim=(-1, -2))
        return E
    def get_output(self, h):
        h = grade_project(h, self.allowed_grades, self.sig)
        W2_h = geometric_product(h, self.W2, self.g)
        return scalar_part(W2_h).sum(dim=-1, keepdim=True)

def run_ablation():
    g = torch.tensor([1.0, 1.0, 1.0]) # Cl(3,0)
    sig = CliffordSignature(g)

    configs = {
        "G0": [0],
        "G01": [0, 1],
        "G02": [0, 2],
        "G012": [0, 1, 2],
        "G0123": [0, 1, 2, 3]
    }

    # Task: classify points (3D)
    x_in = torch.randn(256, 1, 3)
    x_mv = embed_vector(x_in, sig)
    y_target = (torch.norm(x_in, dim=-1) < 1.0).float()

    print("Grade Truncation Ablation Results (Cl(3,0)):")
    for name, grades in configs.items():
        energy = CliffordBilinearEnergy(1, 32, 1, g, grades)
        energy.set_input(x_mv)
        rule = LinearDot()
        engine = EPEngine(energy, rule, n_free=20, n_clamped=10, beta=0.1, dt=0.1)

        start = time.time()
        h_init = torch.zeros(256, 32, 8)
        h_free = engine.free_phase(h_init)
        out = energy.get_output(h_free)
        loss = torch.nn.functional.mse_loss(out, y_target).item()
        end = time.time()

        print(f"Config {name:6}: MSE Loss {loss:.4f}, Time {end-start:.4f}s")

if __name__ == "__main__":
    run_ablation()
