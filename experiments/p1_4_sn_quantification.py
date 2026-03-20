import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product
from cliffeq.training.ep_engine import EPEngine
from cliffordlayers.signature import CliffordSignature
import time

class CliffordBilinearEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_dim, out_nodes, g, use_spectral_norm=False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.g = g
        self.sig = CliffordSignature(g)
        self.hidden_dim = hidden_dim
        # Using nn.Linear to benefit from apply_sn()
        self.w1_lin = nn.Linear(in_nodes * self.sig.n_blades, hidden_dim * self.sig.n_blades, bias=False)
        self.w2_lin = nn.Linear(hidden_dim * self.sig.n_blades, out_nodes * self.sig.n_blades, bias=False)
        self.input_x = None
        self.apply_sn()

    def set_input(self, x):
        self.input_x = x # (B, Nin, I)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Simplified bilinear energy using linear layers
        # E = 0.5 ||h||^2 - <h, W1 x>
        B = h.shape[0]
        h_flat = h.reshape(B, -1)
        x_flat = self.input_x.reshape(B, -1)
        w1x = self.w1_lin(x_flat)
        E = 0.5 * torch.sum(h_flat ** 2, dim=-1) - torch.sum(h_flat * w1x, dim=-1)
        return E

    def get_output(self, h):
        B = h.shape[0]
        h_flat = h.reshape(B, -1)
        out_flat = self.w2_lin(h_flat)
        out = out_flat.reshape(B, -1, self.sig.n_blades)
        return scalar_part(out).sum(dim=-1, keepdim=True)

def run_sn_quantification():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    n_samples = 256
    x_in = torch.randn(n_samples, 1, 2)
    x_mv = embed_vector(x_in, sig)
    y_target = torch.randn(n_samples, 1)

    results = {}
    for sn_on in [False, True]:
        energy = CliffordBilinearEnergy(1, 64, 1, g, use_spectral_norm=sn_on)
        energy.set_input(x_mv)
        rule = LinearDot()
        engine = EPEngine(energy, rule, n_free=20, n_clamped=10, beta=0.5, dt=0.2) # High dt to see stability

        h_init = torch.randn(n_samples, 64, 4)

        start = time.time()
        # Track convergence over 50 steps
        h = h_init.clone()
        energies = []
        for i in range(50):
            with torch.no_grad():
                energies.append(energy(h).sum().item())
            h = rule.step(h, energy, 0.2)
        end = time.time()

        name = "SN_ON" if sn_on else "SN_OFF"
        results[name] = {"energies": energies}
        print(f"Config {name}: Initial Energy {energies[0]:.4f}, Final Energy {energies[-1]:.4f}")

    return results

if __name__ == "__main__":
    run_sn_quantification()
