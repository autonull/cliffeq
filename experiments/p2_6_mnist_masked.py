import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from cliffeq.models.pc import CliffordPC
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
    masked_mv[~mask, 0] = 0.0 # Only mask intensity? Or whole MV?
    # If we mask coordinates too, model has to learn spatial layout.
    # TODO says "masked pixels zeroed out".
    return masked_mv, mask

def rotate_mnist(images, angle):
    from torchvision.transforms.functional import rotate
    return rotate(images, angle)

def run_experiment():
    print("="*80)
    print("P2.6: Clifford Predictive Coding on MNIST Masked Reconstruction")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)

    train_loader, test_loader = get_mnist_subset(n_train=500, n_test=100)

    # Layer dims: 784 (input) -> 512 -> 256
    clifford_dims = [784, 512, 256]
    pc_clifford = CliffordPC(clifford_dims, sig_g).to(device)

    optimizer = torch.optim.Adam(pc_clifford.parameters(), lr=0.001)

    n_epochs = 5

    print("\nTraining Clifford-PC...")
    for epoch in range(n_epochs):
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            mv = encode_mnist_to_clifford(images, sig)
            masked_mv, _ = apply_mask(mv)

            states = pc_clifford(masked_mv, n_iter=10, alpha=0.01)
            preds = pc_clifford.predict_all(states)

            loss = 0
            for l in range(1, len(states)):
                loss += F.mse_loss(states[l-1], preds[l].detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.6f}")

    # Evaluation
    print("\nEvaluation (Reconstruction and Rotation)...")
    pc_clifford.eval()

    angles = [0, 30, 90]
    results = {}

    for angle in angles:
        mse_sum = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                if angle != 0:
                    images = rotate_mnist(images, angle)

                mv = encode_mnist_to_clifford(images, sig)
                masked_mv, mask = apply_mask(mv)

                states = pc_clifford(masked_mv, n_iter=20, alpha=0.01)
                recon_mv = pc_clifford.predict_all(states)[1]
                recon_intensity = recon_mv[..., 0]
                true_intensity = mv[..., 0]

                mse_sum += F.mse_loss(recon_intensity, true_intensity).item()

        avg_mse = mse_sum / len(test_loader)
        results[f"mse_{angle}"] = avg_mse
        print(f"  Angle {angle:2}° MSE: {avg_mse:.6f}")

    # Check non-scalar activity
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        mv = encode_mnist_to_clifford(images.to(device), sig)
        states = pc_clifford(mv, n_iter=20, alpha=0.01)
        preds = pc_clifford.predict_all(states)
        error = states[0] - preds[1]
        print(f"\nNon-scalar error activity (Layer 0 error):")
        for i in range(1, 8):
            print(f"  Blade {i} magnitude: {error[..., i].abs().mean().item():.6f}")

    os.makedirs("results", exist_ok=True)
    with open("results/p2_6_mnist_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_experiment()
