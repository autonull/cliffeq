"""
PV3: Scene Geometry Estimation
Task: Surface normal estimation from synthetic 3D scenes.
Domain: Vision - Geometry estimation with Clifford-EP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from cliffeq.models.flat import EPModel
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot
from cliffeq.algebra.utils import embed_vector, scalar_part
from cliffordlayers.signature import CliffordSignature

def generate_synthetic_scene(n_samples=200, img_size=(32, 32)):
    """
    Generate synthetic depth maps and surface normals for simple geometric primitives.
    """
    H, W = img_size
    xx, yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))

    depths = []
    normals = []

    for _ in range(n_samples):
        # Choose between Plane and Sphere
        if np.random.rand() > 0.5:
            # Plane: ax + by + cz + d = 0 => z = (-ax - by - d) / c
            n = np.random.randn(3)
            n = n / np.linalg.norm(n)
            # Ensure n[2] is positive to be "facing" the camera
            if n[2] < 0: n = -n
            d_val = np.random.uniform(2, 5)
            # z = (d_val - n[0]*x - n[1]*y) / n[2]
            z = (d_val - n[0]*xx - n[1]*yy) / n[2]
            depth = z
            normal = np.tile(n, (H, W, 1))
        else:
            # Sphere: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2
            r = np.random.uniform(1.5, 2.5)
            x0, y0 = np.random.uniform(-0.5, 0.5, 2)
            z0 = np.random.uniform(4, 6)

            # (z-z0)^2 = r^2 - (x-x0)^2 - (y-y0)^2
            disc = r**2 - (xx - x0)**2 - (yy - y0)**2
            mask = disc > 0
            z = np.zeros_like(xx) + z0 + 5 # fallback far
            z[mask] = z0 - np.sqrt(disc[mask])
            depth = z

            # Normal: (x-x0, y-y0, z-z0) normalized
            n = np.zeros((H, W, 3))
            n[..., 0] = xx - x0
            n[..., 1] = yy - y0
            n[..., 2] = z - z0
            norm_val = np.linalg.norm(n, axis=-1, keepdims=True)
            n = - n / (norm_val + 1e-6) # Face camera
            n[~mask] = np.array([0, 0, 1])
            normal = n

        depths.append(depth)
        normals.append(normal)

    return torch.from_numpy(np.array(depths)).float(), torch.from_numpy(np.array(normals)).float()

class SceneGeometryDataset(Dataset):
    def __init__(self, depths, normals):
        self.depths = depths
        self.normals = normals
    def __len__(self):
        return len(self.depths)
    def __getitem__(self, idx):
        # Input: depth map as proxy for image features
        return self.depths[idx].unsqueeze(0), self.normals[idx].permute(2, 0, 1)

class SceneGeometryEPModel(nn.Module):
    def __init__(self, sig_g, hidden_dim=64):
        super().__init__()
        self.sig = CliffordSignature(sig_g)
        # Energy function for per-pixel state
        # For simplicity, we'll treat each pixel as a node in a flat graph
        # and use a BilinearEnergy to predict normals
        self.energy = BilinearEnergy(
            in_nodes=1, # 1 depth value per pixel
            hidden_nodes=1, # 1 multivector state per pixel
            sig_g=sig_g,
            use_spectral_norm=True
        )
        self.ep_model = EPModel(
            energy_fn=self.energy,
            dynamics_rule=LinearDot(),
            n_free=10,
            n_clamped=5,
            beta=0.1,
            dt=0.1
        )

    def forward(self, depth):
        # depth: (B, 1, H, W)
        B, C, H, W = depth.shape
        x = depth.permute(0, 2, 3, 1).reshape(B * H * W, 1)
        # Embed depth as scalar part of Clifford multivector
        x_mv = torch.zeros(B * H * W, 1, self.sig.n_blades, device=depth.device)
        x_mv[..., 0] = x

        self.energy.set_input(x_mv)
        h_init = torch.zeros(B * H * W, 1, self.sig.n_blades, device=depth.device)
        h_free = self.ep_model.engine.free_phase(h_init)

        # Extract normal from vector part (blades 1,2,3)
        normal = h_free[:, 0, 1:4]
        normal = F.normalize(normal, dim=-1)
        return normal.reshape(B, H, W, 3).permute(0, 3, 1, 2)

def angular_error(pred, target):
    # pred, target: (B, 3, H, W)
    dot = torch.sum(pred * target, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)
    angle = torch.acos(dot) * (180.0 / np.pi)
    return angle.mean()

def run_pv3():
    print("=" * 80)
    print("PV3: Scene Geometry Estimation (Surface Normals)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0, 1.0])

    depths, normals = generate_synthetic_scene(300)
    train_loader = DataLoader(SceneGeometryDataset(depths[:250], normals[:250]), batch_size=16, shuffle=True)
    test_loader = DataLoader(SceneGeometryDataset(depths[250:], normals[250:]), batch_size=16)

    model = SceneGeometryEPModel(sig_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for depth, normal in train_loader:
            depth, normal = depth.to(device), normal.to(device)

            # EP training step instead of backprop
            B, C, H, W = depth.shape
            x_depth = depth.permute(0, 2, 3, 1).reshape(B * H * W, 1)
            x_normal = normal.permute(0, 2, 3, 1).reshape(B * H * W, 3)

            # Embed depth as scalar part
            x_mv = torch.zeros(B * H * W, 1, model.sig.n_blades, device=device)
            x_mv[..., 0] = x_depth

            # Loss for EP: normal error
            def normal_loss(h, target):
                # h: (BN, 1, I), target: (BN, 3)
                pred_normal = h[:, 0, 1:4]
                # normalized pred normal
                pred_normal = F.normalize(pred_normal, dim=-1)
                return 0.5 * torch.sum((pred_normal - target) ** 2)

            h_free = model.ep_model.train_step(x_mv, x_normal, optimizer, loss_fn=normal_loss)

            # Track loss for monitoring
            with torch.no_grad():
                pred = F.normalize(h_free[:, 0, 1:4], dim=-1).reshape(B, H, W, 3).permute(0, 3, 1, 2)
                loss = F.mse_loss(pred, normal)
                total_loss += loss.item()
        print(f"  Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")

    # Evaluation
    model.eval()
    errors = []
    with torch.no_grad():
        for depth, normal in test_loader:
            depth, normal = depth.to(device), normal.to(device)
            pred = model(depth)
            err = angular_error(pred, normal)
            errors.append(err.item())

    mean_angle_error = np.mean(errors)
    print(f"\nFinal Mean Angle Error: {mean_angle_error:.2f} degrees")

    # Equivariance test: rotate depth map 90 degrees
    with torch.no_grad():
        depth, normal = next(iter(test_loader))
        depth, normal = depth.to(device), normal.to(device)

        # Original prediction
        pred_orig = model(depth)

        # Rotate input
        depth_rot = torch.rot90(depth, k=1, dims=(2, 3))
        pred_rot = model(depth_rot)

        # Rotate original prediction
        # Note: when rotating the image, the normal vectors themselves must also be rotated
        # 90 deg rotation around Z axis: (nx, ny, nz) -> (-ny, nx, nz)
        pred_orig_rot = torch.rot90(pred_orig, k=1, dims=(2, 3))
        # Modify the vector components
        # Original: [nx, ny, nz] at (y, x)
        # Rotated: [-ny, nx, nz] at (x, -y)
        pred_orig_rot_vec = pred_orig_rot.clone()
        nx = pred_orig_rot[:, 0]
        ny = pred_orig_rot[:, 1]
        pred_orig_rot_vec[:, 0] = -ny
        pred_orig_rot_vec[:, 1] = nx

        equiv_err = torch.norm(pred_rot - pred_orig_rot_vec).item()
        print(f"Equivariance Error (90 deg): {equiv_err:.6f}")

    results = {
        "mean_angle_error": mean_angle_error,
        "equiv_error_90": equiv_err
    }
    os.makedirs("results", exist_ok=True)
    with open("results/pv3_results.json", "w") as f:
        json.dump(results, f)
    print("\n✓ PV3 Complete")

if __name__ == "__main__":
    run_pv3()
