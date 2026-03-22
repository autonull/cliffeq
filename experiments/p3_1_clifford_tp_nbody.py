import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cliffeq.models.tp import CliffordTP
from cliffeq.algebra.utils import reverse, clifford_norm_sq, scalar_part, embed_vector
from cliffordlayers.signature import CliffordSignature
import os
import json
import time

def generate_nbody_system(n_samples=500, n_particles=5, dt=0.01, max_steps=5):
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
    angle = torch.tensor(angle_deg * np.pi / 180.0, device=pos.device, dtype=pos.dtype)
    c, s = torch.cos(angle), torch.sin(angle)

    if axis == 'z':
        R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=pos.dtype, device=pos.device)
    elif axis == 'x':
        R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=pos.dtype, device=pos.device)
    else:  # y
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=pos.dtype, device=pos.device)

    return torch.einsum('...ij,jk->...ik', pos, R)

def run_experiment():
    print("="*80)
    print("P3.1: Clifford Target Propagation on N-Body Dynamics")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)

    n_train = 500
    n_test = 100
    n_particles = 5

    print("Generating N-body trajectories...")
    pos_train, vel_train, targets_train = generate_nbody_system(n_samples=n_train, n_particles=n_particles)
    pos_test, vel_test, targets_test = generate_nbody_system(n_samples=n_test, n_particles=n_particles)

    # Encoding: (B, N, 3) -> (B, N, 8)
    def encode(pos):
        mv = torch.zeros(pos.shape[0], pos.shape[1], 8, device=pos.device)
        mv[..., 1:4] = pos # Vector part
        return mv

    # Architecture
    layer_dims = [n_particles, 64, 64, n_particles]
    tp_model = CliffordTP(layer_dims, sig_g).to(device)
    optimizer = torch.optim.Adam(tp_model.parameters(), lr=0.001)

    print("\nTraining Clifford-TP...")
    n_epochs = 20
    for epoch in range(n_epochs):
        model_input = encode(pos_train[:, -1]).to(device)
        model_target = encode(targets_train).to(device)

        # Forward
        activations = tp_model(model_input)
        # Target Propagation
        targets = tp_model.compute_targets(activations, model_target)

        loss = 0
        for l in range(1, len(activations)):
            loss += F.mse_loss(activations[l], targets[l].detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.6f}")

    # Evaluation
    print("\nEvaluation...")
    tp_model.eval()
    with torch.no_grad():
        test_input = encode(pos_test[:, -1]).to(device)
        test_target = targets_test.to(device)

        activations = tp_model(test_input)
        recon_mv = activations[-1]
        recon_pos = recon_mv[..., 1:4]

        mse = F.mse_loss(recon_pos, test_target).item()
        print(f"  Test MSE: {mse:.6f}")

        # Equivariance test
        pos_rot = rotate_configuration(pos_test[:, -1], 45, 'z').to(device)
        test_input_rot = encode(pos_rot)
        activations_rot = tp_model(test_input_rot)
        recon_pos_rot = activations_rot[-1][..., 1:4]

        recon_pos_rotated = rotate_configuration(recon_pos, 45, 'z')
        equiv_error = torch.norm(recon_pos_rot - recon_pos_rotated) / torch.norm(recon_pos)
        print(f"  Equivariance Error (45 deg): {equiv_error.item():.6f}")

    results = {"test_mse": mse, "equiv_error": equiv_error.item()}
    os.makedirs("results", exist_ok=True)
    with open("results/p3_1_tp_nbody_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_experiment()
