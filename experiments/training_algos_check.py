import torch
from cliffeq.training.chl_engine import CHLEngine
from cliffeq.training.cd_engine import CDEngine
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot

def check_chl():
    print("Checking CHLEngine...")
    sig_g = torch.tensor([1.0, 1.0])
    energy = BilinearEnergy(in_nodes=4, hidden_nodes=8, sig_g=sig_g)
    chl = CHLEngine(energy, LinearDot(), n_pos=10, n_neg=10, dt=0.1)

    x = torch.randn(2, 4, 4)
    target = torch.randn(2, 1, 1) # dummy target for BilinearEnergy.get_output
    energy.set_input(x)

    def loss_fn(h, target):
        return 0.5 * torch.sum((energy.get_output(h) - target)**2)

    h_init = torch.zeros(2, 8, 4)
    h_pos = chl.positive_phase(h_init, target, loss_fn, beta=0.1)
    h_neg = chl.negative_phase(h_init)

    print(f"  h_pos shape: {h_pos.shape}, h_neg shape: {h_neg.shape}")
    chl.parameter_update(h_pos, h_neg)
    print("  CHL parameter update successful.")

def check_cd():
    print("Checking CDEngine...")
    sig_g = torch.tensor([1.0, 1.0])
    energy = BilinearEnergy(in_nodes=4, hidden_nodes=8, sig_g=sig_g)
    cd = CDEngine(energy, n_steps=5, alpha=0.1, noise_std=0.01)

    x = torch.randn(2, 4, 4)
    energy.set_input(x)
    h_pos = torch.randn(2, 8, 4)

    optimizer = torch.optim.Adam(energy.parameters(), lr=0.01)
    loss = cd.train_step(h_pos, optimizer)
    print(f"  CD train step loss: {loss:.4f}")

if __name__ == "__main__":
    check_chl()
    check_cd()
