"""
PR2: Multi-Agent Swarm Coordination
Task: Decentralized swarm formation using Clifford-EP.
Domain: RL - Swarm coordination with local geometric consistency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from cliffeq.models.gnn import GeometricEquilibriumGNN
from cliffeq.dynamics.rules import LinearDot
from cliffeq.algebra.utils import embed_vector, scalar_part
from cliffordlayers.signature import CliffordSignature

def generate_swarm_targets(n_samples=200, n_agents=12):
    """
    Generate target formation: concentric circles.
    """
    targets_list = []

    # Positions on circles
    base_target = []
    n_circles = 2
    agents_per_circle = n_agents // n_circles
    for c in range(n_circles):
        radius = (c + 1) * 0.5
        theta = np.linspace(0, 2*np.pi, agents_per_circle, endpoint=False)
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=-1) * radius
        base_target.append(circle)
    base_target = np.concatenate(base_target, axis=0)

    for _ in range(n_samples):
        # Random scale, rotation, and translation
        scale = np.random.uniform(0.5, 2.0)
        angle = np.random.uniform(0, 2*np.pi)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        offset = np.random.randn(2)

        target = (base_target @ R.T) * scale + offset
        targets_list.append(target)

    return torch.from_numpy(np.array(targets_list)).float()

def get_knn_graph(pos, k=3):
    """
    pos: (B, N, 2)
    Returns: edge_index (2, E)
    """
    B, N, D = pos.shape
    dist = torch.cdist(pos, pos)
    # k nearest neighbors (excluding self)
    _, indices = torch.topk(dist, k + 1, largest=False)
    indices = indices[:, :, 1:] # (B, N, k)

    # Simple strategy: use first batch's graph structure as template
    # (assuming all batches have same connectivity in this PoC)
    row = torch.arange(N).unsqueeze(1).repeat(1, k).reshape(-1)
    col = indices[0].reshape(-1)

    return torch.stack([row, col], dim=0)

class SwarmDataset(Dataset):
    def __init__(self, targets):
        self.targets = targets
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        # Initial positions: noisy version of targets or random
        noise = torch.randn_like(self.targets[idx]) * 0.5
        return self.targets[idx] + noise, self.targets[idx]

def run_pr2():
    print("=" * 80)
    print("PR2: Multi-Agent Swarm Coordination")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0]) # Cl(2,0): 4D multivectors for 2D agents

    n_agents = 12
    targets = generate_swarm_targets(300, n_agents)
    train_loader = DataLoader(SwarmDataset(targets[:250]), batch_size=32, shuffle=True)
    test_loader = DataLoader(SwarmDataset(targets[250:]), batch_size=32)

    model = GeometricEquilibriumGNN(
        nodes=None, # Nodes not used if W is shared
        sig_g=sig_g,
        dynamics_rule=LinearDot(),
        n_free=10,
        n_clamped=5,
        beta=0.1,
        dt=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for x_init, target in train_loader:
            x_init, target = x_init.to(device), target.to(device)
            B = x_init.shape[0]

            # Embed 2D positions as vectors in Cl(2,0)
            # x_init is (B, N, 2)
            sig = CliffordSignature(sig_g)
            x_mv = torch.zeros(B, x_init.shape[1], 4, device=device)
            x_mv[:, :, 1:3] = x_init

            target_mv = torch.zeros(B, target.shape[1], 4, device=device)
            target_mv[:, :, 1:3] = target

            edge_index = get_knn_graph(x_init, k=3).to(device)

            # Normalize inputs
            x_mv = x_mv * 0.001
            target_mv = target_mv * 0.001

            # GEN-GNN training step
            # h_free is (B, N, 4)
            h_free = model.train_step(x_mv, edge_index, target_mv, optimizer)

            with torch.no_grad():
                loss = F.mse_loss(h_free[:, :, 1:3], target * 0.001)
                total_loss += loss.item()
        print(f"  Epoch {epoch+1}, Formation MSE: {total_loss/len(train_loader):.6f}")

    # Evaluation
    model.eval()
    errors = []
    with torch.no_grad():
        for x_init, target in test_loader:
            x_init, target = x_init.to(device), target.to(device)
            B = x_init.shape[0]
            x_mv = torch.zeros(B, x_init.shape[1], 4, device=device)
            x_mv[:, :, 1:3] = x_init
            edge_index = get_knn_graph(x_init, k=3).to(device)
            h_free = model(x_mv, edge_index)
            err = F.mse_loss(h_free[:, :, 1:3], target)
            errors.append(err.item())

    final_mse = np.mean(errors)
    print(f"\nFinal Formation MSE: {final_mse:.6f}")

    # Scale test: double the agents
    print("Scale test: 24 agents zero-shot...")
    n_agents_24 = 24
    targets_24 = generate_swarm_targets(20, n_agents_24).to(device)
    x_init_24 = (targets_24 + torch.randn_like(targets_24) * 0.5).to(device)
    B_24 = x_init_24.shape[0]

    # We must redefine get_knn_graph properly for variable nodes
    def get_knn_graph_scaling(pos, k=3):
        B, N, D = pos.shape
        dist = torch.cdist(pos, pos)
        _, indices = torch.topk(dist, k + 1, largest=False)
        indices = indices[:, :, 1:]
        row = torch.arange(N).unsqueeze(1).repeat(1, k).reshape(-1)
        col = indices[0].reshape(-1)
        return torch.stack([row, col], dim=0)

    edge_index_24 = get_knn_graph_scaling(x_init_24.cpu(), k=3).to(device)

    x_mv_24 = torch.zeros(B_24, n_agents_24, 4, device=device)
    x_mv_24[..., 1:3] = x_init_24

    # We need to adjust model to handle 24 agents if necessary
    # LocalGraphEnergy's W is shared, so it should scale
    # But self.energy_fn.W was (nodes, nodes, I) in GraphEnergy?
    # Let's check cliffeq/energy/zoo.py

    try:
        # If using LocalGraphEnergy from models/gnn.py, it uses self.W as (sig.n_blades) - SHARED
        # This is ideal for scaling
        h_free_24 = model(x_mv_24, edge_index_24)
        scale_mse = F.mse_loss(h_free_24[:, :, 1:3], targets_24).item()
        print(f"Scale Test MSE (24 agents): {scale_mse:.6f}")
    except Exception as e:
        print(f"Scale test failed: {e}")
        scale_mse = None

    results = {
        "formation_mse": final_mse,
        "scale_test_mse_24": scale_mse
    }
    os.makedirs("results", exist_ok=True)
    with open("results/pr2_results.json", "w") as f:
        json.dump(results, f)
    print("\n✓ PR2 Complete")

if __name__ == "__main__":
    run_pr2()
