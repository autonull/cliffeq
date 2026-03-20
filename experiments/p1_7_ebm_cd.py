import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product
from cliffordlayers.signature import CliffordSignature

class CliffordEBMEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_dim, g):
        super().__init__()
        self.g = g
        self.sig = CliffordSignature(g)
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_nodes, self.sig.n_blades) * 0.1)
        self.input_x = None
    def set_input(self, x):
        self.input_x = x
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        E = 0.5 * torch.sum(h ** 2, dim=(-1, -2))
        W1_x = geometric_product(self.input_x, self.W1, self.g)
        E = E - torch.sum(h * W1_x, dim=(-1, -2))
        return E

def run_ebm_cd():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    energy = CliffordEBMEnergy(1, 16, g)

    # Task: classify points (same as P1.1 but with CD)
    x = torch.randn(32, 1, 2)
    x_mv = embed_vector(x, sig)
    energy.set_input(x_mv)

    optimizer = torch.optim.Adam(energy.parameters(), lr=0.01)

    print("Training Clifford EBM with CD-1")
    for epoch in range(10):
        # 1. Positive phase: Use "real" hidden states (clamped to data? for EBM we usually just have x and h)
        # Here we sample h from P(h|x) ≈ exp(-E(h,x))
        # Since E is quadratic in h, h ~ N(W1 x, I)
        with torch.no_grad():
            w1x = geometric_product(x_mv, energy.W1, energy.g)
            h_pos = w1x + torch.randn_like(w1x) * 0.1

        # 2. Negative phase: sample h from model using Langevin
        h_neg = torch.randn_like(h_pos)
        alpha = 0.1
        noise_std = 0.01
        for _ in range(5): # 5 steps of Langevin
            h_neg.requires_grad_(True)
            E = energy(h_neg).sum()
            grad = torch.autograd.grad(E, h_neg)[0]
            h_neg = h_neg - alpha * grad + torch.randn_like(h_neg) * noise_std
            h_neg = h_neg.detach()

        # 3. Update
        optimizer.zero_grad()
        loss = energy(h_pos).mean() - energy(h_neg).mean()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_ebm_cd()
