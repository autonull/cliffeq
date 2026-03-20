import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot, GeomProduct, ExpMap, RotorOnly, Riemannian, GradeSplit, WedgeUpdate
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product
from cliffordlayers.signature import CliffordSignature
import time

class CliffordBilinearEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_dim, out_nodes, g):
        super().__init__()
        self.g = g
        self.sig = CliffordSignature(g)
        self.hidden_dim = hidden_dim
        self.in_nodes = in_nodes
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_nodes, self.sig.n_blades) * 0.1)
        self.input_x = None
    def set_input(self, x):
        self.input_x = x
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, hidden_dim, I)
        # self.input_x: (B, in_nodes, I)
        # self.W1: (hidden_dim, in_nodes, I)
        E = 0.5 * torch.sum(h ** 2, dim=(-1, -2))
        W1_x = geometric_product(self.input_x, self.W1, self.g) # (B, hidden_dim, I)
        E = E - torch.sum(h * W1_x, dim=(-1, -2))
        return E

def run_shootout():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    energy = CliffordBilinearEnergy(1, 16, 1, g)
    x_in = torch.randn(32, 1, 2)
    x_mv = embed_vector(x_in, sig)
    energy.set_input(x_mv)

    h_init = torch.randn(32, 16, 4)
    dt = 0.1
    n_steps = 20

    rules = {
        "LinearDot": LinearDot(),
        "GeomProduct": GeomProduct(g, normalize=True),
        "ExpMap": ExpMap(g),
        "RotorOnly": RotorOnly(g),
        "Riemannian": Riemannian(),
        "GradeSplit": GradeSplit({0: 0.1, 1: 0.2, 2: 0.3}),
        "WedgeUpdate": WedgeUpdate(g)
    }

    print("Update Rule Shootout Results:")
    for name, rule in rules.items():
        h = h_init.clone()
        energies = []
        start = time.time()
        for i in range(n_steps):
            with torch.no_grad():
                energies.append(energy(h).sum().item())
            h = rule.step(h, energy, dt)
        end = time.time()
        print(f"Rule {name:12}: Initial Energy {energies[0]:8.2f}, Final Energy {energies[-1]:8.2f}, Time {end-start:.4f}s")

if __name__ == "__main__":
    run_shootout()
