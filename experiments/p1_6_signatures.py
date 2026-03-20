import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product
from cliffeq.training.ep_engine import EPEngine
from cliffordlayers.signature import CliffordSignature
import time

class CliffordBilinearEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_dim, out_nodes, g):
        super().__init__()
        self.g = g
        self.sig = CliffordSignature(g)
        self.hidden_dim = hidden_dim
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_nodes, self.sig.n_blades) * 0.1)
        self.input_x = None
    def set_input(self, x):
        self.input_x = x
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        E = 0.5 * torch.sum(h ** 2, dim=(-1, -2))
        W1_x = geometric_product(self.input_x, self.W1, self.g)
        E = E - torch.sum(h * W1_x, dim=(-1, -2))
        return E

def run_signatures():
    signatures = {
        "Cl(2,0) Euclidean 2D": torch.tensor([1.0, 1.0]),
        "Cl(3,0) Euclidean 3D": torch.tensor([1.0, 1.0, 1.0]),
        "Cl(1,1) Hyperbolic  ": torch.tensor([1.0, -1.0]),
        "Cl(0,2) Anti-Euclid  ": torch.tensor([-1.0, -1.0]),
    }

    print("Algebra Signature Exploration (EP Convergence):")
    for name, g in signatures.items():
        sig = CliffordSignature(g)
        energy = CliffordBilinearEnergy(1, 16, 1, g)

        # Consistent random input size matching sig.dim
        x_in = torch.randn(32, 1, sig.dim)
        x_mv = embed_vector(x_in, sig)
        energy.set_input(x_mv)

        rule = LinearDot()
        engine = EPEngine(energy, rule, n_free=20, n_clamped=10, beta=0.1, dt=0.1)

        h_init = torch.randn(32, 16, sig.n_blades) * 0.01

        start = time.time()
        h_free = engine.free_phase(h_init)
        end = time.time()

        with torch.no_grad():
            final_e = energy(h_free).sum().item()
            h_norm = torch.norm(h_free).item()

        print(f"Signature {name}: Final Energy {final_e:8.2f}, State Norm {h_norm:6.4f}, Time {end-start:.4f}s")

if __name__ == "__main__":
    run_signatures()
