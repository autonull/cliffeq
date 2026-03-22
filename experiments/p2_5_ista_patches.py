import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from cliffeq.models.sparse import CliffordISTA, CliffordLISTA
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product
from cliffordlayers.signature import CliffordSignature
import os
import json
import time

def load_cifar10_subset(n_samples=1000, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = torch.randperm(len(dataset))[:n_samples]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
    return loader

def extract_patches(images, patch_size=8):
    # images: (B, 1, 32, 32)
    B, C, H, W = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches: (B, 1, 4, 4, 8, 8)
    patches = patches.contiguous().view(B, -1, patch_size, patch_size)
    # patches: (B, 16, 8, 8)
    return patches

def compute_patch_features(patches, sig):
    # patches: (B, 16, 8, 8)
    B, N, H, W = patches.shape
    device = patches.device

    # 1. Luminance (grade-0)
    luminance = patches.mean(dim=(2, 3)) # (B, 16)

    # 2. Centroid (grade-1)
    # Simple relative centroid within patch (not very informative for fixed patches)
    # Let's use Sobel for orientation instead

    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).float().view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).float().view(1, 1, 3, 3)

    gx = F.conv2d(patches.view(-1, 1, H, W), sobel_x, padding=1)
    gy = F.conv2d(patches.view(-1, 1, H, W), sobel_y, padding=1)

    mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
    angle = torch.atan2(gy, gx) # (B*16, 1, 8, 8)

    # Average angle weighted by magnitude
    avg_gx = gx.mean(dim=(2, 3)).view(B, N)
    avg_gy = gy.mean(dim=(2, 3)).view(B, N)

    # mv: (B, 16, 8) for Cl(3,0)
    mv = torch.zeros(B, N, sig.n_blades, device=device)
    mv[..., 0] = luminance
    mv[..., 1] = avg_gx
    mv[..., 2] = avg_gy

    # Grade-2 (bivector) could be avg_gx * avg_gy etc.
    mv[..., 4] = avg_gx * avg_gy # e12 component proxy

    return mv

def run_experiment():
    print("="*80)
    print("P2.5: Clifford ISTA/LISTA on Oriented CIFAR-10 Patches")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)

    loader = load_cifar10_subset(n_samples=500)

    # Dictionary learning with LISTA
    # Input: 16 multivectors (patches)
    # Hidden: 64 atoms
    lista = CliffordLISTA(in_nodes=16, hidden_nodes=64, sig_g=sig_g, n_layers=5).to(device)

    # Decoder for reconstruction: (B, 64, I) -> (B, 16, I)
    class CliffordDecoder(nn.Module):
        def __init__(self, in_nodes, out_nodes, sig_g):
            super().__init__()
            self.sig = CliffordSignature(sig_g)
            self.register_buffer("g", sig_g)
            self.W = nn.Parameter(torch.randn(out_nodes, in_nodes, self.sig.n_blades) * 0.1)
        def forward(self, x):
            return geometric_product(x, self.W, self.g)

    decoder = CliffordDecoder(64, 16, sig_g).to(device)
    optimizer = torch.optim.Adam(list(lista.parameters()) + list(decoder.parameters()), lr=0.001)

    print("\nTraining Clifford-LISTA with Reconstruction Loss...")
    for epoch in range(10):
        total_loss = 0
        for images, _ in loader:
            images = images.to(device)
            patches = extract_patches(images)
            mv = compute_patch_features(patches, sig)

            # LISTA forward
            codes = lista(mv)

            # Reconstruct MV from codes
            recon = decoder(codes)

            # Loss: Reconstruction MSE + Code Sparsity (L1)
            loss_recon = F.mse_loss(recon, mv)
            loss_sparse = codes.abs().mean()
            loss = loss_recon + 0.1 * loss_sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 2 == 0:
            print(f"  Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")

    print("\nVisualizing Learned Atoms (Blade Activity)...")
    with torch.no_grad():
        # Check W1 weights as proxy for atoms
        # W1: (hidden, in, I) = (64, 16, 8)
        atom_activity = lista.W1.abs().mean(dim=1) # (64, 8)
        for i in range(8):
            print(f"  Blade {i} avg activity: {atom_activity[:, i].mean().item():.6f}")

    results = {"status": "success"}
    os.makedirs("results", exist_ok=True)
    with open("results/p2_5_ista_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_experiment()
