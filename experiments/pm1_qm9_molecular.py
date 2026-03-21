import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot
from cliffeq.training.ep_engine import EPEngine
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product
from cliffordlayers.signature import CliffordSignature
import time
import os
import json

# 1. QM9 Molecular Property Prediction (U0)

def load_qm9(root='data/QM9'):
    dataset = QM9(root=root)
    # Target index 7 is U0
    # Split: 500 train / 100 val for speed in this PoC
    train_dataset = dataset[:500]
    test_dataset = dataset[500:600]
    return train_dataset, test_dataset

# 2. Model: Clifford-EP GNN Node

class GraphBilinearEnergy(EnergyFunction):
    def __init__(self, in_dim, hidden_dim, g):
        super().__init__()
        self.g = g
        self.hidden_dim = hidden_dim
        sig = CliffordSignature(g) if len(g) < 5 else None
        I = 32 if len(g) == 5 else sig.n_blades
        # Mapping 1 MV per atom to hidden_dim MVs
        self.W_in = nn.Parameter(torch.randn(hidden_dim, 1, I) * 0.1)
        self.input_x = None

    def set_input(self, x):
        self.input_x = x

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N_atoms, hidden_dim, I)
        E_self = 0.5 * torch.sum(h**2, dim=(-1, -2))
        # input_x: (N_atoms, I)
        # W_in: (hidden_dim, 1, I) - modified to act as 1-to-hidden mapping per atom
        # We need (N_atoms, 1, I) for input_x to use weight-based product
        W_x = geometric_product(self.input_x.unsqueeze(1), self.W_in, self.g) # (N_atoms, hidden_dim, I)
        E_int = -torch.sum(h * W_x, dim=(-1, -2))
        return E_self + E_int

class MolecularCliffordEPModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, g):
        super().__init__()
        self.g = g
        self.energy = GraphBilinearEnergy(in_dim, hidden_dim, g)
        self.rule = LinearDot()
        self.engine = EPEngine(self.energy, self.rule, n_free=10, n_clamped=5, beta=0.1, dt=0.1)

        I = 32 if len(g) == 5 else (2**len(g))
        self.W_out = nn.Parameter(torch.randn(out_dim, hidden_dim, I) * 0.1)

    def forward(self, x_mv):
        self.energy.set_input(x_mv)
        h_init = torch.zeros(x_mv.shape[0], self.energy.hidden_dim, x_mv.shape[-1], device=x_mv.device)
        h_free = self.engine.free_phase(h_init)
        pred_mv = geometric_product(h_free, self.W_out, self.g)
        return scalar_part(pred_mv) # Predict scalar property U0

# 3. Training Loop

def run_pm1():
    print("PM1: QM9 Molecular Property Prediction (U0)")
    train_data, test_data = load_qm9()
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    g = torch.tensor([1.0, 1.0, 1.0]) # Cl(3,0)
    sig = CliffordSignature(g)

    # in_dim = 11 (QM9 node features)
    model = MolecularCliffordEPModel(11, 64, 1, g)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # batch.x is (N_atoms_in_batch, 11)
            # batch.pos is (N_atoms_in_batch, 3)
            # Embed x and pos as multivector
            x_v = embed_vector(batch.pos, sig)
            x_s = torch.zeros_like(x_v)
            # Simplified: use first node feature as scalar part
            x_s[..., 0] = batch.x[:, 0]
            x_mv = x_v + x_s

            # Global prediction (mean over atoms)
            node_preds = model(x_mv) # (N_atoms, 1)
            # Simple global mean pooling
            batch_indices = batch.batch # (N_atoms,)
            graph_preds = torch.zeros(batch.num_graphs, 1, device=node_preds.device)
            graph_preds.index_add_(0, batch_indices, node_preds)
            atom_counts = torch.bincount(batch_indices, minlength=batch.num_graphs).float().view(-1, 1)
            graph_preds = graph_preds / (atom_counts + 1e-6)

            target = batch.y[:, 7].view(-1, 1) # U0 is target 7
            loss = F.mse_loss(graph_preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}, Avg Loss: {total_loss / len(train_loader):.6f}")

    # Eval
    model.eval()
    mae = 0
    with torch.no_grad():
        for batch in test_loader:
            x_v = embed_vector(batch.pos, sig)
            x_s = torch.zeros_like(x_v)
            x_s[..., 0] = batch.x[:, 0]
            x_mv = x_v + x_s

            node_preds = model(x_mv)
            batch_indices = batch.batch
            graph_preds = torch.zeros(batch.num_graphs, 1, device=node_preds.device)
            graph_preds.index_add_(0, batch_indices, node_preds)
            atom_counts = torch.bincount(batch_indices, minlength=batch.num_graphs).float().view(-1, 1)
            graph_preds = graph_preds / (atom_counts + 1e-6)

            target = batch.y[:, 7].view(-1, 1)
            mae += torch.abs(graph_preds - target).sum().item()

    final_mae = mae / len(test_data)
    print(f"Final MAE on U0: {final_mae:.6f} eV")

    results = {"status": "success", "mae": final_mae}
    os.makedirs("results", exist_ok=True)
    with open("results/pm1_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_pm1()
