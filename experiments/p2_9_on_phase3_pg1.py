"""
P2.9 on Phase 3 PG1: Test bottleneck insertion on N-body dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cliffeq.models.bottleneck_v2 import CliffordEPBottleneckV2
from cliffordlayers.signature import CliffordSignature


def generate_nbody_system(n_samples=400, n_particles=5, dt=0.01, max_steps=5):
    """Generate N-body trajectories."""
    positions_list = []
    velocities_list = []
    targets_list = []

    for sample in range(n_samples):
        pos = torch.randn(max_steps, n_particles, 3) * 2.0
        vel = torch.randn(max_steps, n_particles, 3) * 0.1

        for t in range(max_steps - 1):
            acc = torch.zeros(n_particles, 3)
            for i in range(n_particles):
                for j in range(n_particles):
                    if i != j:
                        r = pos[t, j] - pos[t, i]
                        dist = torch.norm(r) + 1e-6
                        acc[i] += r / (dist ** 3 + 1e-6)

            vel[t + 1] = vel[t] + acc * dt
            pos[t + 1] = pos[t] + vel[t + 1] * dt

        positions_list.append(pos)
        velocities_list.append(vel)
        targets_list.append(pos[-1] + vel[-1] * dt)

    return torch.stack(positions_list), torch.stack(velocities_list), torch.stack(targets_list)


def rotate_configuration(pos, angle_deg, axis='z'):
    """Apply rotation to particle configuration."""
    angle = torch.tensor(angle_deg * np.pi / 180.0, device=pos.device, dtype=pos.dtype)
    c, s = torch.cos(angle), torch.sin(angle)

    if axis == 'z':
        R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=pos.dtype, device=pos.device)
    elif axis == 'x':
        R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=pos.dtype, device=pos.device)
    else:  # y
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=pos.dtype, device=pos.device)

    return torch.einsum('...ij,jk->...ik', pos, R)


class SimpleBaselineModel(nn.Module):
    """MLP baseline for N-body."""
    def __init__(self, n_particles=5, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(n_particles * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_particles * 3)
        self.n_particles = n_particles

    def forward(self, pos):
        B, N, _ = pos.shape
        x = pos.view(B, -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        delta = self.fc_out(h)
        return delta.view(B, N, 3) + pos


class BaselineWithBottleneck(nn.Module):
    """MLP with Clifford-EP bottleneck insertion."""
    def __init__(self, n_particles=5, sig_g=None):
        super().__init__()
        self.n_particles = n_particles
        self.fc1 = nn.Linear(n_particles * 3, 64)
        self.bottleneck = CliffordEPBottleneckV2(
            in_dim=64,
            out_dim=32,
            sig_g=sig_g,
            n_ep_steps=2,
            step_size=0.01
        )
        self.fc2 = nn.Linear(32, 32)
        self.fc_out = nn.Linear(32, n_particles * 3)

    def forward(self, pos):
        B, N, _ = pos.shape
        x = pos.view(B, -1)
        h = F.relu(self.fc1(x))
        h = self.bottleneck(h)  # Clifford-EP layer
        h = F.relu(self.fc2(h))
        delta = self.fc_out(h)
        return delta.view(B, N, 3) + pos


def evaluate_equivariance(model, pos_batch, vel_batch, rotation_angles=[15, 45, 90]):
    """Evaluate SO(3) equivariance."""
    device = pos_batch.device
    model.eval()

    if pos_batch.ndim == 4:
        pos_batch = pos_batch[:, -1, :, :]

    violations = []
    with torch.no_grad():
        pred_base = model(pos_batch)

        for angle in rotation_angles:
            for axis in ['x', 'y', 'z']:
                pos_rot = rotate_configuration(pos_batch, angle, axis).to(device)
                pred_rot = model(pos_rot)
                pred_base_rotated = rotate_configuration(pred_base, angle, axis).to(device)
                violation = torch.norm(pred_rot - pred_base_rotated) / (torch.norm(pred_base) + 1e-6)
                violations.append(violation.item())

    return np.mean(violations)


def main():
    print("=" * 70)
    print("P2.9 on Phase 3 PG1: Bottleneck on N-Body Dynamics")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    n_particles = 5

    print("\nGenerating N-body trajectories...")
    positions, velocities, targets = generate_nbody_system(n_samples=400, n_particles=n_particles)

    train_data = TensorDataset(positions[:300].to(device), targets[:300].to(device))
    test_data = TensorDataset(positions[300:].to(device), targets[300:].to(device))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Train baseline
    print("\n" + "-" * 70)
    print("Baseline MLP")
    print("-" * 70)
    baseline = SimpleBaselineModel(n_particles=n_particles).to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01)

    for epoch in range(5):
        for pos_batch, targets_batch in train_loader:
            pos_current = pos_batch[:, -1, :, :]
            preds = baseline(pos_current)
            loss = F.mse_loss(preds, targets_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}: training")

    baseline.eval()
    baseline_test_loss = 0.0
    with torch.no_grad():
        for pos_batch, targets_batch in test_loader:
            pos_current = pos_batch[:, -1, :, :]
            preds = baseline(pos_current)
            baseline_test_loss += F.mse_loss(preds, targets_batch).item()
    baseline_test_loss /= len(test_loader)

    pos_test, targets_test = next(iter(test_loader))
    baseline_equivar = evaluate_equivariance(baseline, pos_test, None)

    print(f"Baseline MSE: {baseline_test_loss:.6f}")
    print(f"Baseline equivariance violation: {baseline_equivar:.6f}")

    # Train with bottleneck
    print("\n" + "-" * 70)
    print("MLP with Clifford-EP Bottleneck")
    print("-" * 70)
    with_bottleneck = BaselineWithBottleneck(n_particles=n_particles, sig_g=sig_g).to(device)
    optimizer = torch.optim.Adam(with_bottleneck.parameters(), lr=0.01)

    for epoch in range(5):
        for pos_batch, targets_batch in train_loader:
            pos_current = pos_batch[:, -1, :, :]
            preds = with_bottleneck(pos_current)
            loss = F.mse_loss(preds, targets_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}: training")

    with_bottleneck.eval()
    bottleneck_test_loss = 0.0
    with torch.no_grad():
        for pos_batch, targets_batch in test_loader:
            pos_current = pos_batch[:, -1, :, :]
            preds = with_bottleneck(pos_current)
            bottleneck_test_loss += F.mse_loss(preds, targets_batch).item()
    bottleneck_test_loss /= len(test_loader)

    bottleneck_equivar = evaluate_equivariance(with_bottleneck, pos_test, None)

    print(f"Bottleneck MSE: {bottleneck_test_loss:.6f}")
    print(f"Bottleneck equivariance violation: {bottleneck_equivar:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: P2.9 Bottleneck on PG1")
    print("=" * 70)
    print(f"{'Metric':<30} {'Baseline':<15} {'Bottleneck':<15}")
    print("-" * 70)
    print(f"{'Test MSE':<30} {baseline_test_loss:<15.6f} {bottleneck_test_loss:<15.6f}")
    print(f"{'Equivariance violation':<30} {baseline_equivar:<15.6f} {bottleneck_equivar:<15.6f}")
    print("-" * 70)

    mse_better = baseline_test_loss < bottleneck_test_loss
    equivar_better = baseline_equivar < bottleneck_equivar

    if not equivar_better:
        improvement = (baseline_equivar - bottleneck_equivar) / (baseline_equivar + 1e-6)
        print(f"\n✓ Bottleneck: {improvement*100:.1f}% better equivariance")
    else:
        degradation = (bottleneck_equivar - baseline_equivar) / (baseline_equivar + 1e-6)
        print(f"\n⚠ Baseline: {degradation*100:.1f}% better equivariance")

    if not mse_better:
        mse_improvement = (baseline_test_loss - bottleneck_test_loss) / (baseline_test_loss + 1e-6)
        print(f"✓ Bottleneck: {mse_improvement*100:.1f}% better MSE")
    else:
        mse_degradation = (bottleneck_test_loss - baseline_test_loss) / (baseline_test_loss + 1e-6)
        print(f"⚠ Baseline: {mse_degradation*100:.1f}% better MSE")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
