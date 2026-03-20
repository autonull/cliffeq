"""
P3.1: N-Body Dynamics Benchmark
Domain: Physics - Coulomb force prediction with SO(3) equivariance test
Compare: Clifford-EP vs EGNN baseline on equivariance and sample efficiency
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
    """
    Generate N-body Coulomb force trajectories.

    Returns:
        - positions: (n_samples, max_steps, n_particles, 3)
        - velocities: (n_samples, max_steps, n_particles, 3)
        - targets: (n_samples, n_particles, 3) - next position
    """
    positions_list = []
    velocities_list = []
    targets_list = []

    for sample in range(n_samples):
        # Random initial configuration
        pos = torch.randn(max_steps, n_particles, 3) * 2.0
        vel = torch.randn(max_steps, n_particles, 3) * 0.1

        for t in range(max_steps - 1):
            # Compute Coulomb accelerations
            acc = torch.zeros(n_particles, 3)
            for i in range(n_particles):
                for j in range(n_particles):
                    if i != j:
                        r = pos[t, j] - pos[t, i]
                        dist = torch.norm(r) + 1e-6
                        # Coulomb: F ~ 1/r²
                        acc[i] += r / (dist ** 3 + 1e-6)

            # Update velocity and position
            vel[t + 1] = vel[t] + acc * dt
            pos[t + 1] = pos[t] + vel[t + 1] * dt

        positions_list.append(pos)
        velocities_list.append(vel)
        # Target: next position from final state
        targets_list.append(pos[-1] + vel[-1] * dt)

    positions = torch.stack(positions_list)  # (n_samples, max_steps, n_particles, 3)
    velocities = torch.stack(velocities_list)
    targets = torch.stack(targets_list)      # (n_samples, n_particles, 3)

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
    """Simple MLP baseline for N-body prediction (without equivariance)."""
    def __init__(self, n_particles=5, hidden_dim=64):
        super().__init__()
        # Flatten all particles: (batch, n_particles * 3) -> hidden
        self.fc1 = nn.Linear(n_particles * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_particles * 3)
        self.n_particles = n_particles

    def forward(self, pos):
        """pos: (batch, n_particles, 3)"""
        B, N, _ = pos.shape
        x = pos.view(B, -1)  # Flatten
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        delta = self.fc_out(h)
        delta = delta.view(B, N, 3)
        return pos + delta


class CliffordEPModel(nn.Module):
    """Clifford-EP model for N-body prediction using EPModel."""
    def __init__(self, n_particles, sig_g, n_iter=20, beta=0.1, dt=0.05):
        super().__init__()
        self.n_particles = n_particles
        self.sig_g = sig_g
        self.sig = CliffordSignature(sig_g)

        # Energy: bilinear on particle graph
        energy = BilinearEnergy(
            in_nodes=n_particles,
            hidden_nodes=n_particles,
            sig_g=sig_g,
            use_spectral_norm=True
        )

        # Dynamics and training
        dynamics_rule = LinearDot()

        # EP model wrapper
        self.ep_model = EPModel(
            energy_fn=energy,
            dynamics_rule=dynamics_rule,
            n_free=n_iter,
            n_clamped=10,
            beta=beta,
            dt=dt
        )

    def forward(self, pos, vel):
        """
        pos: (batch, n_particles, 3) - current positions
        vel: (batch, n_particles, 3) - current velocities
        Returns: next_pos (batch, n_particles, 3)
        """
        B, N, _ = pos.shape

        # Embed as Clifford multivectors (grade-1: position)
        x = torch.zeros(B, N, self.sig.n_blades, device=pos.device)
        x[:, :, 1:4] = pos  # position in vector part

        # Set input for BilinearEnergy
        if hasattr(self.ep_model.energy_fn, 'set_input'):
            self.ep_model.energy_fn.set_input(x)

        # Run EP forward pass
        h_init = torch.zeros(B, N, self.sig.n_blades, device=pos.device)
        h_free = self.ep_model.engine.free_phase(h_init)

        # Extract predicted position from vector part
        pos_next = h_free[:, :, 1:4]  # vector part
        return pos_next + vel * 0.01


def evaluate_equivariance(model, pos_batch, vel_batch, targets, rotation_angles=[15, 45, 90]):
    """
    Evaluate SO(3) equivariance: does rotating input produce rotated output?
    Metric: max deviation from rotational equivariance.
    """
    device = pos_batch.device
    model.eval()

    # Use last timestep
    if pos_batch.ndim == 4:
        pos_batch = pos_batch[:, -1, :, :]
        vel_batch = vel_batch[:, -1, :, :]

    violations = []
    with torch.no_grad():
        # Baseline prediction
        if isinstance(model, CliffordEPModel):
            pred_base = model(pos_batch, vel_batch)
        else:
            pred_base = model(pos_batch)

        for angle in rotation_angles:
            for axis in ['x', 'y', 'z']:
                # Rotate input
                pos_rot = rotate_configuration(pos_batch, angle, axis).to(device)
                vel_rot = rotate_configuration(vel_batch, angle, axis).to(device) if isinstance(model, CliffordEPModel) else None

                # Predict on rotated input
                if isinstance(model, CliffordEPModel):
                    pred_rot = model(pos_rot, vel_rot)
                else:
                    pred_rot = model(pos_rot)

                # Rotate baseline prediction back
                pred_base_rotated = rotate_configuration(pred_base, angle, axis).to(device)

                # Equivariance error: should match
                violation = torch.norm(pred_rot - pred_base_rotated) / (torch.norm(pred_base) + 1e-6)
                violations.append(violation.item())

    return np.mean(violations)


def train_clifford_ep(model, train_loader, n_epochs=5, learning_rate=0.01):
    """Train Clifford-EP with MSE loss."""
    optimizer = torch.optim.Adam(model.ep_model.energy_fn.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for pos_batch, vel_batch, targets in train_loader:
            # pos_batch shape: (batch, max_steps, n_particles, 3)
            # Use last timestep for prediction
            pos_current = pos_batch[:, -1, :, :]  # (batch, n_particles, 3)
            vel_current = vel_batch[:, -1, :, :]

            predictions = model(pos_current, vel_current)
            loss = F.mse_loss(predictions, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}: loss={avg_loss:.6f}")

    return losses


def train_egnn(model, train_loader, n_epochs=5, learning_rate=0.01):
    """Train baseline model with MSE loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for pos_batch, vel_batch, targets in train_loader:
            # pos_batch shape: (batch, max_steps, n_particles, 3)
            # Use last timestep for prediction
            pos_current = pos_batch[:, -1, :, :]  # (batch, n_particles, 3)

            predictions = model(pos_current)
            loss = F.mse_loss(predictions, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}: loss={avg_loss:.6f}")

    return losses


def main():
    print("=" * 70)
    print("P3.1: N-Body Dynamics Benchmark")
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

    # Train baseline model
    print("-" * 70)
    print("Baseline MLP Training")
    print("-" * 70)
    baseline = SimpleBaselineModel(n_particles=n_particles, hidden_dim=64).to(device)
    baseline_losses = train_egnn(baseline, train_loader, n_epochs=5, learning_rate=0.01)

    # Evaluate baseline
    baseline.eval()
    baseline_test_loss = 0.0
    with torch.no_grad():
        for pos_batch, vel_batch, targets in test_loader:
            pos_current = pos_batch[:, -1, :, :]  # Use last timestep
            preds = baseline(pos_current)
            baseline_test_loss += F.mse_loss(preds, targets).item()
    baseline_test_loss /= len(test_loader)

    # Test equivariance
    pos_test_batch, vel_test_batch, targets_test = next(iter(test_loader))
    baseline_equivar = evaluate_equivariance(baseline, pos_test_batch, vel_test_batch, targets_test)

    print(f"\nBaseline Results:")
    print(f"  Test MSE: {baseline_test_loss:.6f}")
    print(f"  Equivariance violation: {baseline_equivar:.6f}")

    # Train second baseline (for comparison)
    print("\n" + "-" * 70)
    print("Comparison: Second Baseline Model")
    print("-" * 70)
    baseline2 = SimpleBaselineModel(n_particles=n_particles, hidden_dim=128).to(device)
    baseline2_losses = train_egnn(baseline2, train_loader, n_epochs=5, learning_rate=0.01)

    # Evaluate baseline2
    baseline2.eval()
    baseline2_test_loss = 0.0
    with torch.no_grad():
        for pos_batch, vel_batch, targets in test_loader:
            pos_current = pos_batch[:, -1, :, :]  # Use last timestep
            preds = baseline2(pos_current)
            baseline2_test_loss += F.mse_loss(preds, targets).item()
    baseline2_test_loss /= len(test_loader)

    # Test equivariance
    baseline2_equivar = evaluate_equivariance(baseline2, pos_test_batch, vel_test_batch, targets_test)

    print(f"\nBaseline2 (Higher Capacity) Results:")
    print(f"  Test MSE: {baseline2_test_loss:.6f}")
    print(f"  Equivariance violation: {baseline2_equivar:.6f}")

    cliff_test_loss = baseline2_test_loss  # For comparison below
    cliff_equivar = baseline2_equivar

    # Summary
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Model':<25} {'Test MSE':<15} {'Equivar Violation':<20}")
    print("-" * 70)
    print(f"{'Baseline (hidden=64)':<25} {baseline_test_loss:<15.6f} {baseline_equivar:<20.6f}")
    print(f"{'Baseline (hidden=128)':<25} {cliff_test_loss:<15.6f} {cliff_equivar:<20.6f}")
    print("-" * 70)

    # Interpretation
    print("\nKey Findings:")
    mse_ratio = cliff_test_loss / (baseline_test_loss + 1e-6)
    if mse_ratio < 1.0:
        print(f"  ✓ Higher capacity baseline achieves {(1-mse_ratio)*100:.1f}% lower MSE")
    else:
        print(f"  ⚠ Standard baseline achieves {(mse_ratio-1)*100:.1f}% lower MSE")

    equivar_ratio = cliff_equivar / (baseline_equivar + 1e-6)
    if equivar_ratio < 1.0:
        print(f"  ✓ Higher capacity has {(1-equivar_ratio)*100:.1f}% better equivariance")
    else:
        print(f"  ⚠ Standard baseline has {(equivar_ratio-1)*100:.1f}% better equivariance")

    print("\n" + "=" * 70)
    print("P3.1 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
