import torch
from cliffeq.models.rotor import RotorEPModel
from cliffeq.models.gnn import GeometricEquilibriumGNN
from cliffeq.models.hybrid import CliffordEPBottleneck
from cliffeq.energy.zoo import NormEnergy, BilinearEnergy
from cliffeq.dynamics.rules import LinearDot

def check_rotor():
    print("Checking RotorEPModel...")
    # E = 0.5 * ||q||^2
    energy = NormEnergy()
    model = RotorEPModel(energy, n_free=10, n_clamped=0, beta=0.0, dt=0.1)
    x = torch.randn(2, 10) # dummy input
    q_init = torch.zeros(2, 1, 4)
    q_init[..., 0] = 1.0
    out = model(x, q_init)
    print(f"  Rotor output shape: {out.shape}")
    # Rotor should stay normalized
    norm = torch.norm(out, dim=-1)
    print(f"  Rotor norm: {norm}")

def check_gnn():
    print("Checking GeometricEquilibriumGNN...")
    sig_g = torch.tensor([1.0, 1.0])
    model = GeometricEquilibriumGNN(nodes=4, sig_g=sig_g, dynamics_rule=LinearDot(), n_free=10, n_clamped=0, beta=0.0, dt=0.1)
    x_init = torch.randn(2, 4, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    out = model(x_init, edge_index)
    print(f"  GNN output shape: {out.shape}")

def check_bottleneck():
    print("Checking CliffordEPBottleneck...")
    sig_g = torch.tensor([1.0, 1.0])
    energy = BilinearEnergy(in_nodes=4, hidden_nodes=4, sig_g=sig_g)
    model = CliffordEPBottleneck(energy, LinearDot(), n_free=5, dt=0.1, comp=4)
    x = torch.randn(2, 16) # 4 nodes * 4 blades
    out = model(x)
    print(f"  Bottleneck output shape: {out.shape}")

if __name__ == "__main__":
    check_rotor()
    check_gnn()
    check_bottleneck()
