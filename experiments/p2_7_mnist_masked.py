import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from cliffeq.models.tp import CliffordTP
from cliffeq.algebra.utils import reverse, clifford_norm_sq, scalar_part
from cliffordlayers.signature import CliffordSignature
import os
import json
import time

def get_mnist_subset(n_train=500, n_test=100, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_indices = torch.randperm(len(train_dataset))[:n_train]
    test_indices = torch.randperm(len(test_dataset))[:n_test]

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def encode_mnist_to_clifford(images, sig):
    # images: (B, 1, 28, 28)
    B, _, H, W = images.shape
    device = images.device
    pixels = images.view(B, -1)

    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    mv = torch.zeros(B, H*W, sig.n_blades, device=device)
    mv[..., 0] = pixels
    mv[..., 1] = grid_x
    mv[..., 2] = grid_y

    return mv

def apply_mask(mv, mask_ratio=0.5):
    B, N, I = mv.shape
    mask = torch.rand(B, N, device=mv.device) > mask_ratio
    masked_mv = mv.clone()
    masked_mv[~mask] = 0.0
    return masked_mv, mask

class ScalarTP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(layer_dims[i], layer_dims[i-1]) * 0.01)
            for i in range(1, len(layer_dims))
        ])

    def forward(self, x):
        activations = [x]
        for w in self.weights:
            y = F.linear(activations[-1], w)
            activations.append(y)
        return activations

    def compute_targets(self, activations, global_target):
        targets = [None] * len(activations)
        targets[-1] = global_target

        for l in range(len(self.weights) - 1, -1, -1):
            w = self.weights[l]
            # Pseudoinverse approximation: W^-1 ≈ W^T / ||W||^2
            n2 = (w**2).mean() * w.shape[1]
            w_pinv = w.t() / (n2 + 1e-8)
            targets[l] = F.linear(targets[l+1], w_pinv)

        return targets

def run_experiment():
    print("="*80)
    print("P2.7: Clifford Target Propagation on MNIST Masked Reconstruction")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)

    train_loader, test_loader = get_mnist_subset(n_train=500, n_test=100)

    # Layer dims: 784 -> 128 -> 128 -> 784
    # Simpler architecture for stability
    clifford_dims = [784, 128, 128, 784]
    tp_clifford = CliffordTP(clifford_dims, sig_g).to(device)
    # Initialize with very small weights for stability in linear TP
    for p in tp_clifford.parameters():
        p.data.mul_(0.01)

    scalar_dims = [784*8, 128*8, 128*8, 784*8]
    tp_scalar = ScalarTP(scalar_dims).to(device)
    for p in tp_scalar.parameters():
        p.data.mul_(0.01)

    opt_clifford = torch.optim.Adam(tp_clifford.parameters(), lr=0.0001)
    opt_scalar = torch.optim.Adam(tp_scalar.parameters(), lr=0.001)

    n_epochs = 5

    print("\nTraining Clifford-TP...")
    for epoch in range(n_epochs):
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            mv = encode_mnist_to_clifford(images, sig)
            masked_mv, _ = apply_mask(mv)

            activations = tp_clifford(masked_mv)
            targets = tp_clifford.compute_targets(activations, mv)

            loss = 0
            for l in range(1, len(activations)):
                loss += F.mse_loss(activations[l], targets[l].detach())

            opt_clifford.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tp_clifford.parameters(), 1.0)
            opt_clifford.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.6f}")

    print("\nTraining Scalar-TP...")
    for epoch in range(n_epochs):
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            mv = encode_mnist_to_clifford(images, sig)
            masked_mv, _ = apply_mask(mv)

            x_scalar = masked_mv.view(images.shape[0], -1)
            y_scalar = mv.view(images.shape[0], -1)

            activations = tp_scalar(x_scalar)
            targets = tp_scalar.compute_targets(activations, y_scalar)

            loss = 0
            for l in range(1, len(activations)):
                loss += F.mse_loss(activations[l], targets[l].detach())

            opt_scalar.zero_grad()
            loss.backward()
            opt_scalar.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.6f}")

    print("\nEvaluation...")
    tp_clifford.eval()
    tp_scalar.eval()

    mse_clifford = 0
    mse_scalar = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            mv = encode_mnist_to_clifford(images, sig)
            masked_mv, mask = apply_mask(mv)

            activations = tp_clifford(masked_mv)
            recon_mv = activations[-1]
            recon_intensity = recon_mv[..., 0]
            true_intensity = mv[..., 0]
            mse_clifford += F.mse_loss(recon_intensity, true_intensity).item()

            x_scalar = masked_mv.view(images.shape[0], -1)
            y_scalar = mv.view(images.shape[0], -1)
            activations_s = tp_scalar(x_scalar)
            recon_s = activations_s[-1].view(mv.shape)
            recon_s_intensity = recon_s[..., 0]
            mse_scalar += F.mse_loss(recon_s_intensity, true_intensity).item()

    print(f"Final Test MSE (Intensity):")
    print(f"  Clifford-TP: {mse_clifford/len(test_loader):.6f}")
    print(f"  Scalar-TP:   {mse_scalar/len(test_loader):.6f}")

    results = {
        "clifford_tp_mse": mse_clifford/len(test_loader),
        "scalar_tp_mse": mse_scalar/len(test_loader)
    }
    os.makedirs("results", exist_ok=True)
    with open("results/p2_7_mnist_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_experiment()
