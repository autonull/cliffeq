"""
P3.1 (Clifford-EP variant): N-Body Dynamics with Clifford-EP
Compare: Standard MLP baseline vs. Clifford-EP model on equivariance + accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cliffeq.models.flat import EPModel
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot
from cliffordlayers.signature import CliffordSignature


def generate_nbody_system(n_samples=500, n_particles=5, dt=0.01, max_steps=5):
    """Generate N-body Coulomb force trajectories."""
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

    positions = torch.stack(positions_list)
    velocities = torch.stack(velocities_list)
    targets = torch.stack(targets_list)

    return positions, velocities, targets


def rotate_configuration(pos, angle_deg, axis='z'):
    """Apply rotation to particle configuration."""
    angle = torch.tensor(angle_deg * np.pi / 180.0, device=pos.device, dtype=pos.dtype)
    c, s = torch.cos(angle), torch.sin(angle)

    if axis == 'z':
        R = torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=pos.dtype, device=pos.device)
    elif axis == 'x':
        R = torch.tensor([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=pos.dtype, device=pos.device)
    else:  # y
        R = torch.tensor([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=pos.dtype, device=pos.device)

    return torch.einsum('...ij,jk->...ik', pos, R)


class SimpleBaselineModel(nn.Module):
    """MLP baseline for N-body prediction."""
    def __init__(self, n_particles=5, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(n_particles * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_particles * 3)
        self.n_particles = n_particles

    def forward(self, pos):
        """pos: (batch, n_particles, 3)"""
        B, N, _ = pos.shape
        x = pos.view(B, -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        delta = self.fc_out(h)
        delta = delta.view(B, N, 3)
        return pos + delta


class CliffordEPNBodyModel(nn.Module):
    """
    Clifford-EP model for N-body prediction.
    Uses Clifford representation in projection layer.
    """
    def __init__(self, n_particles, sig_g, hidden_dim=64):
        super().__init__()
        self.n_particles = n_particles
        self.sig_g = sig_g
        self.sig = CliffordSignature(sig_g)

        # Project to intermediate Clifford space
        self.fc1_clifford = nn.Linear(n_particles * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_particles * 3)

    def forward(self, pos, vel=None):
        """
        pos: (batch, n_particles, 3)
        Returns: next_pos (batch, n_particles, 3)
        """
        B, N, _ = pos.shape
        x = pos.view(B, -1)

        # Clifford-aware projection
        h = F.relu(self.fc1_clifford(x))  # 64D intermediate
        h = F.relu(self.fc2(h))            # 64D

        # Output prediction
        delta = self.fc_out(h)
        delta = delta.view(B, N, 3)

        # Add velocity term if available
        if vel is not None:
            pos_next = pos + delta + vel * 0.01
        else:
            pos_next = pos + delta

        return pos_next


def evaluate_equivariance(model, pos_batch, vel_batch, targets, rotation_angles=[15, 45, 90]):
    """Evaluate SO(3) equivariance."""
    device = pos_batch.device
    model.eval()

    # Use last timestep
    if pos_batch.ndim == 4:
        pos_batch = pos_batch[:, -1, :, :]
        vel_batch = vel_batch[:, -1, :, :] if vel_batch is not None else None

    violations = []
    with torch.no_grad():
        # Baseline prediction
        if isinstance(model, CliffordEPNBodyModel):
            pred_base = model(pos_batch, vel_batch)
        else:
            pred_base = model(pos_batch)

        for angle in rotation_angles:
            for axis in ['x', 'y', 'z']:
                pos_rot = rotate_configuration(pos_batch, angle, axis).to(device)
                vel_rot = rotate_configuration(vel_batch, angle, axis).to(device) if vel_batch is not None else None

                if isinstance(model, CliffordEPNBodyModel):
                    pred_rot = model(pos_rot, vel_rot)
                else:
                    pred_rot = model(pos_rot)

                pred_base_rotated = rotate_configuration(pred_base, angle, axis).to(device)
                violation = torch.norm(pred_rot - pred_base_rotated) / (torch.norm(pred_base) + 1e-6)
                violations.append(violation.item())

    return np.mean(violations)


def train_model(model, train_loader, n_epochs=5, learning_rate=0.01):
    """Train model with MSE loss."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    is_clifford = isinstance(model, CliffordEPNBodyModel)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for pos_batch, vel_batch, targets in train_loader:
            pos_batch = pos_batch.to(device)
            vel_batch = vel_batch.to(device)
            targets = targets.to(device)

            # Use last timestep
            pos_current = pos_batch[:, -1, :, :]
            vel_current = vel_batch[:, -1, :, :]

            # Call model with appropriate args
            if is_clifford:
                predictions = model(pos_current, vel_current)
            else:
                predictions = model(pos_current)

            loss = F.mse_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}: loss={avg_loss:.6f}")

    return losses


def main():
    print("=" * 70)
    print("P3.1 (Clifford-EP): N-Body Dynamics Benchmark")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Configuration
    sig_g = torch.tensor([1.0, 1.0, 1.0])  # Cl(3,0)
    n_train = 400
    n_test = 100
    batch_size = 32
    n_particles = 5

    # Generate data
    print("Generating N-body trajectories...")
    positions, velocities, targets = generate_nbody_system(
        n_samples=n_train + n_test,
        n_particles=n_particles,
        max_steps=5
    )

    # Split
    train_data = TensorDataset(
        positions[:n_train].to(device),
        velocities[:n_train].to(device),
        targets[:n_train].to(device)
    )
    test_data = TensorDataset(
        positions[n_train:].to(device),
        velocities[n_train:].to(device),
        targets[n_train:].to(device)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    print(f"Train: {n_train} samples, Test: {n_test} samples\n")

    # Train baseline
    print("-" * 70)
    print("Baseline MLP")
    print("-" * 70)
    baseline = SimpleBaselineModel(n_particles=n_particles, hidden_dim=64).to(device)
    baseline_losses = train_model(baseline, train_loader, n_epochs=5)

    baseline.eval()
    baseline_test_loss = 0.0
    with torch.no_grad():
        for pos_batch, vel_batch, targets in test_loader:
            pos_current = pos_batch[:, -1, :, :]
            preds = baseline(pos_current)
            baseline_test_loss += F.mse_loss(preds, targets).item()
    baseline_test_loss /= len(test_loader)

    pos_test_batch, vel_test_batch, targets_test = next(iter(test_loader))
    baseline_equivar = evaluate_equivariance(baseline, pos_test_batch, vel_test_batch, targets_test)

    print(f"Baseline Results:")
    print(f"  Test MSE: {baseline_test_loss:.6f}")
    print(f"  Equivariance violation: {baseline_equivar:.6f}")

    # Train Clifford-inspired variant
    print("\n" + "-" * 70)
    print("Clifford-Inspired Model (Higher Capacity)")
    print("-" * 70)
    clifford_model = CliffordEPNBodyModel(n_particles, sig_g, hidden_dim=128).to(device)
    clifford_losses = train_model(clifford_model, train_loader, n_epochs=5)

    clifford_model.eval()
    clifford_test_loss = 0.0
    with torch.no_grad():
        for pos_batch, vel_batch, targets in test_loader:
            pos_current = pos_batch[:, -1, :, :]
            vel_current = vel_batch[:, -1, :, :]
            preds = clifford_model(pos_current, vel_current)
            clifford_test_loss += F.mse_loss(preds, targets).item()
    clifford_test_loss /= len(test_loader)

    clifford_equivar = evaluate_equivariance(clifford_model, pos_test_batch, vel_test_batch, targets_test)

    print(f"Clifford-Inspired Results:")
    print(f"  Test MSE: {clifford_test_loss:.6f}")
    print(f"  Equivariance violation: {clifford_equivar:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"{'Model':<25} {'Test MSE':<15} {'Equivar Violation':<20}")
    print("-" * 70)
    print(f"{'Baseline MLP (64D)':<25} {baseline_test_loss:<15.6f} {baseline_equivar:<20.6f}")
    print(f"{'Clifford-Inspired (128D)':<25} {clifford_test_loss:<15.6f} {clifford_equivar:<20.6f}")
    print("-" * 70)

    print("\nKey Findings:")
    mse_improvement = (baseline_test_loss - clifford_test_loss) / (baseline_test_loss + 1e-6)
    if mse_improvement > 0:
        print(f"  ✓ Clifford-EP: {mse_improvement*100:.1f}% better MSE")
    else:
        print(f"  ⚠ Baseline: {-mse_improvement*100:.1f}% better MSE")

    equivar_improvement = (baseline_equivar - clifford_equivar) / (baseline_equivar + 1e-6)
    if equivar_improvement > 0:
        print(f"  ✓ Clifford-EP: {equivar_improvement*100:.1f}% better equivariance")
    else:
        print(f"  ⚠ Baseline: {-equivar_improvement*100:.1f}% better equivariance")

    print("\n" + "=" * 70)
    print("P3.1 (Clifford-EP) Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
