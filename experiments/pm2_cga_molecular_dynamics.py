import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot
from cliffeq.training.ep_engine import EPEngine
from cliffeq.algebra.utils import geometric_product, scalar_part
import numpy as np
import json
import os

# PM2: CGA Molecular Dynamics (Rigid-Body Trajectory Prediction)

def generate_rigid_body_trajectories(n_samples=1000, n_steps=50):
    # Simulate a rigid tetrahedron (4 atoms)
    # Target: predict state at t+5 from t=0..4
    # For PoC, simplified linear + angular motion
    trajectories = []
    for _ in range(n_samples):
        pos = np.random.randn(4, 3) # Initial relative positions (tetrahedron)
        center = np.random.randn(3)
        v = np.random.randn(3) * 0.1
        omega = np.random.randn(3) * 0.1 # angular velocity

        traj = []
        for t in range(n_steps):
            dt = t * 0.1
            # Translation
            curr_center = center + v * dt
            # Rotation (simplified)
            theta = np.linalg.norm(omega) * dt
            if theta > 0:
                axis = omega / np.linalg.norm(omega)
                # Rodrigues rotation
                # x' = x cos theta + (axis x x) sin theta + axis (axis . x) (1 - cos theta)
                # Using a simple rotation for now
                traj.append(curr_center + pos)
            else:
                traj.append(curr_center + pos)
        trajectories.append(np.array(traj))
    return torch.from_numpy(np.array(trajectories)).float()

class CGAMotorEnergy(EnergyFunction):
    def __init__(self, hidden_dim):
        super().__init__()
        self.g = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0])
        # W must be (hidden_dim, Nin, 32)
        # We input the past 5 steps as motors (or positions)
        self.W_in = nn.Parameter(torch.randn(hidden_dim, 5, 32) * 0.1)
        self.input_x = None

    def set_input(self, x):
        self.input_x = x

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        E_self = 0.5 * torch.sum(h**2, dim=(-1, -2))
        W_x = geometric_product(self.input_x, self.W_in, self.g)
        E_int = -torch.sum(h * W_x, dim=(-1, -2))
        return E_self + E_int

class CGAMotorModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.energy = CGAMotorEnergy(hidden_dim)
        self.rule = LinearDot()
        self.engine = EPEngine(self.energy, self.rule, n_free=10, n_clamped=5, beta=0.1, dt=0.1)
        self.W_out = nn.Parameter(torch.randn(1, hidden_dim, 32) * 0.1)
        self.g = self.energy.g

    def forward(self, x_mv):
        # x_mv: (B, 5, 32) - 5 past steps
        self.energy.set_input(x_mv)
        B = x_mv.shape[0]
        h_init = torch.zeros(B, self.energy.W_in.shape[0], 32, device=x_mv.device)
        h_free = self.engine.free_phase(h_init)
        pred_mv = geometric_product(h_free, self.W_out, self.g)
        return pred_mv # (B, 1, 32)

def run_pm2():
    print("PM2: CGA Molecular Dynamics (Rigid-Body Trajectory Prediction)")
    data = generate_rigid_body_trajectories(1200)
    # data is (1200, 50, 4, 3) -> 1200 samples, 50 steps, 4 atoms, 3D
    # Use center of mass for simplicity
    centers = data.mean(dim=2) # (1200, 50, 3)

    # Task: predict center at t=10 from t=0..4
    X = centers[:, 0:5, :] # (1200, 5, 3)
    y = centers[:, 10, :]  # (1200, 3)

    X_train, y_train = X[:1000], y[:1000]
    X_test, y_test = X[1000:], y[1000:]

    model = CGAMotorModel(64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # CGA Point Embedding: X = x + 0.5|x|^2 e_inf + e_o
    # e_inf = e4 + e5, e_o = 0.5(e5 - e4)
    def embed_cga_point(x):
        B, L, _ = x.shape
        res = torch.zeros(B, L, 32, device=x.device)
        res[..., 1:4] = x
        sq_norm = torch.sum(x**2, dim=-1)
        # e4 is index 4, e5 is index 5
        # e_inf = e4 + e5
        res[..., 4] += 0.5 * sq_norm # component of e4
        res[..., 5] += 0.5 * sq_norm # component of e5
        # e_o = 0.5 * e5 - 0.5 * e4
        res[..., 5] += 0.5
        res[..., 4] -= 0.5
        return res

    X_train_mv = embed_cga_point(X_train)
    X_test_mv = embed_cga_point(X_test)

    print("Training...")
    for epoch in range(10):
        preds_mv = model(X_train_mv) # (B, 1, 32)
        # Extract predicted point coordinates from index 1:4
        preds = preds_mv[:, 0, 1:4]
        loss = F.mse_loss(preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.6f}")

    # Eval
    with torch.no_grad():
        preds_mv = model(X_test_mv)
        preds = preds_mv[:, 0, 1:4]
        test_mse = F.mse_loss(preds, y_test).item()
        print(f"Final Test MSE: {test_mse:.6f}")

    return {"mse": test_mse}

if __name__ == "__main__":
    results = run_pm2()
    with open("results/pm2_results.json", "w") as f:
        json.dump(results, f)
