import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot
from cliffeq.algebra.utils import embed_vector, scalar_part
from cliffeq.training.ep_engine import EPEngine
from cliffordlayers.signature import CliffordSignature
import time

class CliffordBilinearEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_dim, out_nodes, g, use_spectral_norm=False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.g = g
        self.sig = CliffordSignature(g)
        self.hidden_dim = hidden_dim
        # Use nn.Linear to apply spectral norm to weights
        self.w1_lin = nn.Linear(in_nodes * self.sig.n_blades, hidden_dim * self.sig.n_blades, bias=False)
        self.input_x = None
        self.apply_sn()
    def set_input(self, x):
        self.input_x = x
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        h_flat = h.reshape(B, -1)
        x_flat = self.input_x.reshape(B, -1)
        w1x = self.w1_lin(x_flat)
        return 0.5 * torch.sum(h_flat ** 2, dim=-1) - torch.sum(h_flat * w1x, dim=-1)

def run_sn_quantification():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    n_samples = 256
    x_in = torch.randn(n_samples, 1, 2)
    x_mv = embed_vector(x_in, sig)

    print("Spectral Normalization Impact (Cl(2,0)):")
    for beta in [0.01, 0.1, 0.5]:
        print(f"\nNudge Strength beta={beta}:")
        for sn_on in [False, True]:
            energy = CliffordBilinearEnergy(1, 64, 1, g, use_spectral_norm=sn_on)
            energy.set_input(x_mv)
            rule = LinearDot()

            # Track stability over 50 steps
            h = torch.randn(n_samples, 64, 4)
            dt = 0.5 # Aggressive step size to test stability

            energies = []
            diverged = False
            for i in range(50):
                with torch.no_grad():
                    e = energy(h).sum().item()
                    energies.append(e)
                if abs(e) > 1e10 or torch.isnan(h).any():
                    diverged = True
                    break
                h = rule.step(h, energy, dt)

            status = "DIVERGED" if diverged else f"CONVERGED (Final E: {energies[-1]:.2f})"
            name = "SN_ON " if sn_on else "SN_OFF"
            print(f"  {name}: {status}")

if __name__ == "__main__":
    run_sn_quantification()
