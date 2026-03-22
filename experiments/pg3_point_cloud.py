"""
PG3: 3D Point Cloud Classification
Task: Classify synthetic 3D shapes using hierarchical Clifford-EP GEN-GNN.
Domain: Physics/Geometry - Point cloud processing.
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

def knn_graph_fallback(x, k, batch=None):
    # x: (N, 3)
    dist = torch.cdist(x, x)
    _, indices = torch.topk(dist, k + 1, largest=False)
    indices = indices[:, 1:] # (N, k)
    row = torch.arange(x.size(0), device=x.device).unsqueeze(1).repeat(1, k).flatten()
    col = indices.flatten()
    return torch.stack([row, col], dim=0)

def generate_synthetic_3d_shapes(n_samples=400, n_points=128):
    """
    Generate synthetic 3D shapes: Cube, Sphere, Cylinder, Pyramid.
    """
    X = []
    y = []

    for _ in range(n_samples):
        label = np.random.randint(0, 4)
        y.append(label)

        points = []
        if label == 0: # Cube
            points = np.random.uniform(-1, 1, (n_points, 3))
            # Snap to faces
            face = np.random.randint(0, 6, n_points)
            for i in range(n_points):
                dim = face[i] // 2
                side = 1 if face[i] % 2 == 0 else -1
                points[i, dim] = side
        elif label == 1: # Sphere
            points = np.random.randn(n_points, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif label == 2: # Cylinder
            theta = np.random.uniform(0, 2*np.pi, n_points)
            h = np.random.uniform(-1, 1, n_points)
            points = np.stack([np.cos(theta), np.sin(theta), h], axis=-1)
        else: # Pyramid
            points = np.random.uniform(0, 1, (n_points, 3))
            for i in range(n_points):
                points[i, 0] = (points[i, 0] - 0.5) * (1 - points[i, 2])
                points[i, 1] = (points[i, 1] - 0.5) * (1 - points[i, 2])
                points[i, 2] = points[i, 2] - 0.5

        # Add random rotation and noise
        angle = np.random.uniform(0, 2*np.pi)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        # Simple rotation about Z for PoC
        R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        points = points @ R.T
        points += np.random.randn(n_points, 3) * 0.05

        X.append(points)

    return torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).long()

class PointCloudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HierarchicalGENGNN(nn.Module):
    def __init__(self, sig_g, n_classes=4):
        super().__init__()
        self.sig = CliffordSignature(sig_g)
        self.n_blades = self.sig.n_blades

        # Layer 1 GNN
        self.gnn1 = GeometricEquilibriumGNN(None, sig_g, LinearDot(), n_free=5, n_clamped=0, beta=0.0, dt=0.1)
        # Layer 2 GNN
        self.gnn2 = GeometricEquilibriumGNN(None, sig_g, LinearDot(), n_free=5, n_clamped=0, beta=0.0, dt=0.1)

        # Readout
        self.fc = nn.Linear(self.n_blades, n_classes)

    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        device = x.device

        # Embed points as Clifford vectors
        x_mv = torch.zeros(B, N, self.n_blades, device=device)
        x_mv[:, :, 1:4] = x

        # Layer 1: k-NN Graph
        # For simplicity, use the same graph for all batches
        edge_index1 = knn_graph_fallback(x[0], k=8)

        # EP relaxation Layer 1
        h1 = self.gnn1(x_mv, edge_index1)

        # Hierarchical Pooling (crude: mean pool for PoC)
        # A proper hierarchical GEN-GNN would use FPS and another GNN layer
        # Here we do global mean pool and a second GEN-GNN step on the aggregate
        h_pool = h1.mean(dim=1, keepdim=True) # (B, 1, I)

        # Layer 2: Self-loop graph for the aggregate
        edge_index2 = torch.tensor([[0], [0]], device=device)
        h2 = self.gnn2(h_pool, edge_index2) # (B, 1, I)

        # Readout
        logits = self.fc(h2.squeeze(1))
        return logits

def run_pg3():
    print("=" * 80)
    print("PG3: 3D Point Cloud Classification")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0, 1.0])

    X, y = generate_synthetic_3d_shapes(500)
    train_loader = DataLoader(PointCloudDataset(X[:400], y[:400]), batch_size=32, shuffle=True)
    test_loader = DataLoader(PointCloudDataset(X[400:], y[400:]), batch_size=32)

    model = HierarchicalGENGNN(sig_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("Training (Backprop through EP iterations)...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == by).sum().item()

        print(f"  Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Acc: {correct/400:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            correct += (logits.argmax(dim=1) == by).sum().item()

    test_acc = correct / 100
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # Rotation equivariance test
    print("Testing SO(2) rotation equivariance...")
    with torch.no_grad():
        bx, by = next(iter(test_loader))
        bx = bx.to(device)

        # Original prediction
        logits_orig = model(bx)

        # Rotate input 90 deg about Z
        angle = np.pi / 2
        R = torch.tensor([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], device=device).float()
        bx_rot = bx @ R.T
        logits_rot = model(bx_rot)

        # Logits should be invariant for classification
        equiv_err = torch.norm(F.softmax(logits_orig, dim=1) - F.softmax(logits_rot, dim=1)).item()
        print(f"Equivariance Error (90 deg Z-rot): {equiv_err:.6f}")

    results = {
        "test_accuracy": test_acc,
        "equiv_error_90z": equiv_err
    }
    os.makedirs("results", exist_ok=True)
    with open("results/pg3_results.json", "w") as f:
        json.dump(results, f)
    print("\n✓ PG3 Complete")

if __name__ == "__main__":
    run_pg3()
