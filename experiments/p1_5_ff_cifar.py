import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from cliffeq.training.ff_engine import FFEngine
from cliffeq.algebra.utils import embed_vector, clifford_norm_sq
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
import numpy as np
import json
import os

def load_cifar_subset(n_samples=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = torch.randperm(len(dataset))[:n_samples]
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

def overlay_label(x, y):
    x_flat = x.view(x.shape[0], -1)
    label_info = torch.zeros(x.shape[0], 10, device=x.device)
    label_info[range(x.shape[0]), y] = x_flat.norm(dim=1)
    return torch.cat([x_flat, label_info], dim=1)

def overlay_label_mv(x_mv, y):
    # x_mv: (B, N, I)
    B = x_mv.shape[0]
    # Label as a separate node
    label_node = torch.zeros(B, 1, x_mv.shape[-1], device=x_mv.device)
    # Put label index in one of the slots
    label_node[range(B), 0, (y % 4).long()] = 1.0 # simplistic
    return torch.cat([x_mv, label_node], dim=1)

def run_ff_cifar():
    print("P1.5: Forward-Forward + Clifford on CIFAR-10")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_cifar_subset(2000)
    train_size = 1600
    train_data = [dataset[i] for i in range(train_size)]
    test_data = [dataset[i] for i in range(train_size, 2000)]

    X_train = torch.stack([d[0] for d in train_data])
    y_train = torch.tensor([d[1] for d in train_data])
    X_test = torch.stack([d[0] for d in test_data])
    y_test = torch.tensor([d[1] for d in test_data])

    # 1. Scalar FF
    print("\n--- Scalar FF ---")
    in_dim = 3072 + 10
    scalar_layers = nn.ModuleList([nn.Linear(in_dim, 512), nn.Linear(512, 512)]).to(device)
    def scalar_goodness(h): return h.pow(2).mean(1)
    scalar_engine = FFEngine(scalar_goodness, threshold_theta=10.0)

    h_pos = overlay_label(X_train, y_train).to(device)
    h_neg = overlay_label(X_train, torch.randint(0, 10, y_train.shape)).to(device)
    for layer in scalar_layers:
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        h_pos, h_neg = scalar_engine.train_layer(layer, h_pos, h_neg, optimizer, n_epochs=20)
        h_pos = h_pos / (h_pos.norm(2, 1, keepdim=True) + 1e-4)
        h_neg = h_neg / (h_neg.norm(2, 1, keepdim=True) + 1e-4)

    def predict_scalar(x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_label(x, torch.full((x.shape[0],), label, dtype=torch.long)).to(device)
            goodness = 0
            for layer in scalar_layers:
                h = F.relu(layer(h))
                goodness += scalar_goodness(h)
                h = h / (h.norm(2, 1, keepdim=True) + 1e-4)
            goodness_per_label.append(goodness.unsqueeze(1))
        return torch.cat(goodness_per_label, 1).argmax(1)

    acc = (predict_scalar(X_test) == y_test.to(device)).float().mean().item()
    print(f"  Scalar FF Accuracy: {acc:.4f}")

    # 2. Clifford FF (A)
    print("\n--- Clifford FF (A) ---")
    sig_g = [1, 1, 1]
    sig = CliffordSignature(sig_g)
    # CIFAR: 3*32*32 = 3072. Use nodes of 3 features (RGB)
    # 3072 / 3 = 1024 nodes
    clif_layers = nn.ModuleList([CliffordLinear(sig_g, 1024+1, 64), CliffordLinear(sig_g, 64, 64)]).to(device)
    def clif_goodness(h): return clifford_norm_sq(h, sig).mean(1)
    clif_engine = FFEngine(clif_goodness, threshold_theta=10.0)

    X_train_mv = embed_vector(X_train.permute(0, 2, 3, 1).reshape(train_size, 1024, 3), sig, sig_g).to(device)
    h_pos_mv = overlay_label_mv(X_train_mv, y_train.to(device))
    h_neg_mv = overlay_label_mv(X_train_mv, torch.randint(0, 10, y_train.shape, device=device))

    for layer in clif_layers:
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        h_pos_mv, h_neg_mv = clif_engine.train_layer(layer, h_pos_mv, h_neg_mv, optimizer, n_epochs=20)
        h_pos_mv = h_pos_mv / (torch.norm(h_pos_mv, dim=-1, keepdim=True) + 1e-4)
        h_neg_mv = h_neg_mv / (torch.norm(h_neg_mv, dim=-1, keepdim=True) + 1e-4)

    def predict_clif(x):
        x_mv = embed_vector(x.permute(0, 2, 3, 1).reshape(x.shape[0], 1024, 3), sig, sig_g).to(device)
        goodness_per_label = []
        for label in range(10):
            h = overlay_label_mv(x_mv, torch.full((x.shape[0],), label, device=device))
            goodness = 0
            for layer in clif_layers:
                h = layer(h)
                goodness += clif_goodness(h)
                h = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-4)
            goodness_per_label.append(goodness.unsqueeze(1))
        return torch.cat(goodness_per_label, 1).argmax(1)

    acc_clif = (predict_clif(X_test) == y_test.to(device)).float().mean().item()
    print(f"  Clifford FF Accuracy: {acc_clif:.4f}")

    return {"scalar": acc, "clifford": acc_clif}

if __name__ == "__main__":
    results = run_ff_cifar()
    with open("results/p1_5_ff_cifar.json", "w") as f:
        json.dump(results, f)
