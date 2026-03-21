import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cliffeq.training.ff_engine import FFEngine
from cliffeq.algebra.utils import embed_vector, clifford_norm_sq, scalar_part, geometric_product
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
import time

def generate_p1_1_data(n_samples=2000):
    # Points inside unit circle (class 0) vs inside rotated ellipse (class 1)
    X = torch.randn(n_samples, 2)
    mask0 = torch.norm(X, dim=1) < 1.0
    theta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    X_rot = X @ R.T
    mask1 = (X_rot[:, 0] / 2.0)**2 + (X_rot[:, 1] / 0.5)**2 < 1.0

    n0 = mask0.sum().item()
    n1 = mask1.sum().item()
    n_use = min(int(n0), int(n1), n_samples // 2)
    X_bal = torch.cat([X[mask0][:n_use], X[mask1][:n_use]], dim=0)
    y_bal = torch.cat([torch.zeros(n_use), torch.ones(n_use)], dim=0).long()

    idx = torch.randperm(X_bal.shape[0])
    print(f"Generated {len(idx)} samples: {n0} circle, {n1} ellipse")
    return X_bal[idx], y_bal[idx]

def overlay_label(x, y):
    """Overlay label on data for FF algorithm."""
    B = x.shape[0]
    label_info = torch.zeros(B, 2, device=x.device)
    label_info[range(B), y.long()] = 1.0 # x.norm(dim=1)
    return torch.cat([x, label_info], dim=1)

def overlay_label_mv(x_mv, y):
    B = x_mv.shape[0]
    label_mv = torch.zeros(B, 1, 4, device=x_mv.device)
    label_mv[range(B), 0, y.long() + 1] = 1.0 # use vector parts for label
    return torch.cat([x_mv, label_mv], dim=1)

def run_ff_experiment():
    print("P1.5: Forward-Forward + Clifford Baseline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_p1_1_data(2000)
    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    print(f"Split: {len(X_train)} train, {len(X_test)} test")

    sig_g = [1, 1]
    sig = CliffordSignature(sig_g)

    # 1. Scalar FF
    print("\n--- Scalar FF ---")
    scalar_layers = nn.ModuleList([
        nn.Linear(4, 64),
        nn.Linear(64, 64)
    ]).to(device)

    def scalar_goodness(h):
        return h.pow(2).mean(1)

    scalar_engine = FFEngine(scalar_goodness, threshold_theta=2.0)

    pos_data = overlay_label(X_train, y_train).to(device)
    neg_data = overlay_label(X_train, 1 - y_train).to(device)

    h_pos, h_neg = pos_data, neg_data
    for i, layer in enumerate(scalar_layers):
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.03)
        h_pos, h_neg = scalar_engine.train_layer(layer, h_pos, h_neg, optimizer, n_epochs=50)
        # Re-normalize activations between layers
        h_pos = h_pos / (h_pos.norm(2, 1, keepdim=True) + 1e-4)
        h_neg = h_neg / (h_neg.norm(2, 1, keepdim=True) + 1e-4)
        print(f"  Layer {i+1} trained")

    def predict_scalar(x):
        if x.shape[0] == 0: return torch.tensor([], device=device)
        goodness_per_label = []
        for label in range(2):
            h = overlay_label(x, torch.full((x.shape[0],), label)).to(device)
            goodness = 0
            for layer in scalar_layers:
                h = F.relu(layer(h))
                goodness += scalar_goodness(h)
                h = h / (h.norm(2, 1, keepdim=True) + 1e-4)
            goodness_per_label.append(goodness.unsqueeze(1))
        return torch.cat(goodness_per_label, 1).argmax(1)

    acc = (predict_scalar(X_test) == y_test.to(device)).float().mean().item()
    print(f"  Scalar FF Accuracy: {acc:.4f}")

    # 2. Clifford FF (Variant A: Clifford Norm Goodness)
    print("\n--- Clifford FF (Variant A: Clifford Norm) ---")
    clif_layers = nn.ModuleList([
        CliffordLinear(sig_g, 2, 16),
        CliffordLinear(sig_g, 16, 16)
    ]).to(device)

    def clif_goodness_a(h):
        return clifford_norm_sq(h, sig).mean(1)

    clif_engine_a = FFEngine(clif_goodness_a, threshold_theta=2.0)
    X_train_mv = embed_vector(X_train.unsqueeze(1), sig).to(device)

    pos_data_mv = overlay_label_mv(X_train_mv, y_train.to(device))
    neg_data_mv = overlay_label_mv(X_train_mv, (1 - y_train).to(device))

    h_pos, h_neg = pos_data_mv, neg_data_mv
    for i, layer in enumerate(clif_layers):
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.03)
        h_pos, h_neg = clif_engine_a.train_layer(layer, h_pos, h_neg, optimizer, n_epochs=50)
        h_pos = h_pos / (torch.norm(h_pos, dim=-1, keepdim=True) + 1e-4)
        h_neg = h_neg / (torch.norm(h_neg, dim=-1, keepdim=True) + 1e-4)
        print(f"  Layer {i+1} trained")

    def predict_clif_a(x):
        if x.shape[0] == 0: return torch.tensor([], device=device)
        x_mv = embed_vector(x.unsqueeze(1), sig).to(device)
        goodness_per_label = []
        for label in range(2):
            h = overlay_label_mv(x_mv, torch.full((x.shape[0],), label, device=device))
            goodness = 0
            for layer in clif_layers:
                h = layer(h)
                goodness += clif_goodness_a(h)
                h = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-4)
            goodness_per_label.append(goodness.unsqueeze(1))
        return torch.cat(goodness_per_label, 1).argmax(1)

    acc_a = (predict_clif_a(X_test) == y_test.to(device)).float().mean().item()
    print(f"  Clifford FF (A) Accuracy: {acc_a:.4f}")

    # 3. Clifford FF (Variant B: Learnable Geometric Goodness)
    print("\n--- Clifford FF (Variant B: Learnable Geometric) ---")
    clif_layers_b = nn.ModuleList([
        CliffordLinear(sig_g, 2, 16),
        CliffordLinear(sig_g, 16, 16)
    ]).to(device)

    class LearnableGoodness(nn.Module):
        def __init__(self, n_nodes):
            super().__init__()
            self.w = nn.Parameter(torch.ones(n_nodes))
        def forward(self, h):
            norms = clifford_norm_sq(h, sig)
            return (norms * F.softplus(self.w)).mean(1)

    goodness_modules = nn.ModuleList([LearnableGoodness(16), LearnableGoodness(16)]).to(device)

    h_pos, h_neg = pos_data_mv, neg_data_mv
    for i, layer in enumerate(clif_layers_b):
        optimizer = torch.optim.Adam(list(layer.parameters()) + list(goodness_modules[i].parameters()), lr=0.03)
        engine_b = FFEngine(goodness_modules[i], threshold_theta=2.0)
        h_pos, h_neg = engine_b.train_layer(layer, h_pos, h_neg, optimizer, n_epochs=50)
        h_pos = h_pos / (torch.norm(h_pos, dim=-1, keepdim=True) + 1e-4)
        h_neg = h_neg / (torch.norm(h_neg, dim=-1, keepdim=True) + 1e-4)
        print(f"  Layer {i+1} trained")

    def predict_clif_b(x):
        if x.shape[0] == 0: return torch.tensor([], device=device)
        x_mv = embed_vector(x.unsqueeze(1), sig).to(device)
        goodness_per_label = []
        for label in range(2):
            h = overlay_label_mv(x_mv, torch.full((x.shape[0],), label, device=device))
            goodness = 0
            for i, layer in enumerate(clif_layers_b):
                h = layer(h)
                goodness += goodness_modules[i](h)
                h = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-4)
            goodness_per_label.append(goodness.unsqueeze(1))
        return torch.cat(goodness_per_label, 1).argmax(1)

    acc_b = (predict_clif_b(X_test) == y_test.to(device)).float().mean().item()
    print(f"  Clifford FF (B) Accuracy: {acc_b:.4f}")

if __name__ == "__main__":
    run_ff_experiment()
