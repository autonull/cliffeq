import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from cliffeq.models.pc import CliffordPC
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product
from cliffordlayers.signature import CliffordSignature
import os
import json
import time

def load_qm9_subset(n_samples=500, batch_size=32):
    dataset = QM9(root='data/QM9')
    indices = torch.randperm(len(dataset))[:n_samples]
    loader = DataLoader(dataset[indices], batch_size=batch_size, shuffle=True)
    return loader

class MolecularCliffordPCModel(nn.Module):
    def __init__(self, hidden_dims, sig_g):
        super().__init__()
        self.sig_g = sig_g
        self.sig = CliffordSignature(sig_g)

        # PC needs fixed layer dimensions
        self.pc = CliffordPC(hidden_dims, sig_g)

        # Readout from last layer's scalar part
        self.readout = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_mv, n_iter=10):
        # x_mv: (B, N, I)
        states = self.pc(x_mv, n_iter=n_iter)
        # last layer state
        h_last = states[-1]
        h_scalar = scalar_part(h_last)
        return self.readout(h_scalar), states

def run_experiment():
    print("="*80)
    print("PM1: Clifford Predictive Coding on QM9 (U0 prediction)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)

    loader = load_qm9_subset(n_samples=500)

    # Layer dims: 16 (input nodes) -> 16 -> 16
    # Let's use a fixed number of nodes for PC simplicity
    n_nodes = 16
    model = MolecularCliffordPCModel([n_nodes, 16, 16], sig_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining Clifford-PC on QM9...")
    for epoch in range(10):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            x_v = embed_vector(batch.pos, sig)
            x_s = torch.zeros_like(x_v)
            x_s[..., 0] = batch.x[:, 0]
            x_mv_nodes = x_v + x_s

            B = batch.num_graphs
            x_mv = torch.zeros(B, n_nodes, 8, device=device)

            for i in range(B):
                mask = (batch.batch == i)
                num = min(mask.sum().item(), n_nodes)
                x_mv[i, :num] = x_mv_nodes[mask][:num]

            pred, states = model(x_mv)
            target = batch.y[:, 7].view(-1, 1)

            loss_task = F.mse_loss(pred, target)

            preds_internal = model.pc.predict_all(states)
            loss_pc = 0
            for l in range(1, len(states)):
                loss_pc += F.mse_loss(states[l-1], preds_internal[l].detach())

            loss = loss_task + 0.1 * loss_pc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 2 == 0:
            print(f"  Epoch {epoch+1}, Avg Loss: {total_loss/len(loader):.6f}")

    # Evaluation
    model.eval()
    mae = 0
    with torch.no_grad():
        test_loader = load_qm9_subset(n_samples=100)
        for batch in test_loader:
            batch = batch.to(device)
            x_v = embed_vector(batch.pos, sig)
            x_s = torch.zeros_like(x_v)
            x_s[..., 0] = batch.x[:, 0]
            x_mv_nodes = x_v + x_s

            B = batch.num_graphs
            x_mv = torch.zeros(B, n_nodes, 8, device=device)
            for i in range(B):
                mask = (batch.batch == i)
                num = min(mask.sum().item(), n_nodes)
                x_mv[i, :num] = x_mv_nodes[mask][:num]

            pred, _ = model(x_mv)
            target = batch.y[:, 7].view(-1, 1)
            mae += torch.abs(pred - target).sum().item()

    print(f"\nFinal MAE on U0: {mae/100:.6f} eV")

    results = {"mae": mae/100}
    os.makedirs("results", exist_ok=True)
    with open("results/pm1_pc_qm9_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_experiment()
