import torch
from cliffeq.dynamics.rules import GeomProduct
from cliffeq.energy.base import EnergyFunction
from cliffeq.algebra.utils import embed_vector, geometric_product
from cliffordlayers.signature import CliffordSignature

class SimpleBilinear(EnergyFunction):
    def __init__(self, g):
        super().__init__()
        self.g = g
        self.W = torch.randn(1, 1, 4) * 0.1
        self.input_x = None
    def set_input(self, x):
        self.input_x = x
    def forward(self, h):
        W_x = geometric_product(self.input_x, self.W, self.g)
        return 0.5 * torch.sum(h ** 2) - torch.sum(h * W_x)

def debug_nan():
    g = torch.tensor([1.0, 1.0])
    energy = SimpleBilinear(g)
    x = embed_vector(torch.randn(1, 1, 2), CliffordSignature(g))
    energy.set_input(x)

    rule = GeomProduct(g)
    h = torch.randn(1, 1, 4)
    alpha = 0.1

    print("Initial h:", h)
    for i in range(5):
        h = rule.step(h, energy, alpha)
        print(f"Step {i}, h norm: {torch.norm(h).item()}, max: {h.max().item()}, min: {h.min().item()}")
        if torch.isnan(h).any():
            print("NaN detected!")
            break

if __name__ == "__main__":
    debug_nan()
