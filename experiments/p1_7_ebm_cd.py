import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product
from cliffeq.dynamics.rules import LinearDot
from cliffeq.training.ep_engine import EPEngine
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

def run_ebm_cd_vs_ep():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)

    x = torch.randn(64, 1, 2)
    x_mv = embed_vector(x, sig)
    y_target = (torch.norm(x, dim=-1, keepdim=True) < 1.0).float()

    print("Clifford Energy Training: CD vs EP Comparison")

    for mode in ["CD", "EP"]:
        energy = CliffordEBMEnergy(1, 16, g)
        energy.set_input(x_mv)
        optimizer = torch.optim.Adam(energy.parameters(), lr=0.01)

        losses = []
        for epoch in range(20):
            if mode == "CD":
                with torch.no_grad():
                    w1x = geometric_product(x_mv, energy.W1, energy.g)
                    h_pos = w1x + torch.randn_like(w1x) * 0.1

                h_neg = torch.randn_like(h_pos)
                alpha = 0.1
                noise_std = 0.05
                for _ in range(5):
                    h_neg.requires_grad_(True)
                    E = energy(h_neg).sum()
                    grad = torch.autograd.grad(E, h_neg)[0]
                    h_neg = (h_neg - alpha * grad + torch.randn_like(h_neg) * noise_std).detach()

                optimizer.zero_grad()
                loss = (energy(h_pos).mean() - energy(h_neg).mean())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            else: # EP
                engine = EPEngine(energy, LinearDot(), n_free=10, n_clamped=5, beta=0.1, dt=0.1)
                h_init = torch.zeros(64, 16, 4)

                h_free = engine.free_phase(h_init)
                def loss_fn(h, target):
                    # scalar part mean as simple prediction
                    pred = scalar_part(h).mean(dim=-1, keepdim=True)
                    return 0.5 * torch.sum((pred - target) ** 2)

                h_clamped = engine.clamped_phase(h_free, y_target, loss_fn)
                optimizer.zero_grad()
                engine.parameter_update(h_free, h_clamped)
                optimizer.step()

                with torch.no_grad():
                    cur_loss = loss_fn(h_free, y_target).item()
                losses.append(cur_loss)

        print(f"Mode {mode:2}: Initial Loss {losses[0]:8.4f}, Final Loss {losses[-1]:8.4f}")

if __name__ == "__main__":
    run_ebm_cd_vs_ep()
