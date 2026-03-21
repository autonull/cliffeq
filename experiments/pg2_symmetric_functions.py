import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot, GeomProduct
from cliffeq.training.ep_engine import EPEngine
from cliffeq.algebra.utils import embed_vector, scalar_part, geometric_product, clifford_norm_sq
from cliffeq.benchmarks.metrics import equivariance_violation
from cliffordlayers.signature import CliffordSignature
import time
import numpy as np
from scipy.spatial import ConvexHull
import json

# 1. Tasks: Volume, Force, Zn symmetry, Time-Reversal

def generate_convex_hull_data(n_samples=1000, n_points=20):
    X = torch.randn(n_samples, n_points, 3)
    y = torch.zeros(n_samples, 1)
    for i in range(n_samples):
        hull = ConvexHull(X[i].numpy())
        y[i] = hull.volume
    return X, y

def generate_force_field_data(n_samples=1000, n_charges=10):
    X = torch.randn(n_samples, n_charges, 3)
    q = torch.randn(n_samples, n_charges, 1)
    r = X
    dist = torch.norm(r, dim=-1, keepdim=True)
    F = torch.sum(q * r / (dist**3 + 1e-6), dim=1)
    return X, q, F

def generate_zn_data(n_samples=1200, n_points=12, n=4):
    X = torch.randn(n_samples, n_points, 2)
    y = torch.randint(0, 2, (n_samples, 1))
    for i in range(n_samples):
        if y[i] == 1:
            base_points = torch.randn(n_points // n, 2)
            all_points = []
            for j in range(n):
                theta = 2 * np.pi * j / n
                c, s = np.cos(theta), np.sin(theta)
                R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
                all_points.append(torch.einsum('ij,pj->pi', R, base_points))
            X[i] = torch.cat(all_points, dim=0)
    return X, y.float()

def generate_time_reversal_data(n_samples=1200, n_events=5):
    X = torch.randn(n_samples, n_events, 4)
    y = torch.randint(0, 2, (n_samples, 1))
    for i in range(n_samples):
        if y[i] == 1:
            for j in range(n_events):
                t = X[i, j, 0]
                X[i, j, 1:] = X[i, j, 1:] * (1 - t/10)
        else:
            for j in range(n_events):
                t = X[i, j, 0]
                X[i, j, 1:] = X[i, j, 1:] * (1 + t/10)
    return X, y.float()

# 2. Models: Scalar EP, Clifford-EP

class BilinearEnergy(EnergyFunction):
    def __init__(self, in_dim, hidden_dim, g):
        super().__init__()
        self.g = g
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        sig = CliffordSignature(g) if len(g) < 5 and len(g) != 4 else None
        if len(g) == 5: I = 32
        elif len(g) == 4: I = 16
        else: I = sig.n_blades
        self.W_in = nn.Parameter(torch.randn(hidden_dim, in_dim, I) * 0.1)
        self.input_x = None

    def set_input(self, x):
        self.input_x = x

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        E_self = 0.5 * torch.sum(h**2, dim=(-1, -2))
        W_x = geometric_product(self.input_x, self.W_in, self.g)
        E_int = -torch.sum(h * W_x, dim=(-1, -2))
        return E_self + E_int

class CliffordEPModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, g, task_type='invariant'):
        super().__init__()
        self.g = g
        self.task_type = task_type
        self.energy = BilinearEnergy(in_dim, hidden_dim, g)
        self.rule = LinearDot()
        self.engine = EPEngine(self.energy, self.rule, n_free=20, n_clamped=10, beta=0.1, dt=0.1)

        if len(g) == 5: I = 32
        elif len(g) == 4: I = 16
        else: I = 2**len(g)
        self.W_out = nn.Parameter(torch.randn(out_dim, hidden_dim, I) * 0.1)

    def forward(self, x_mv):
        self.energy.set_input(x_mv)
        B = x_mv.shape[0]
        h_init = torch.zeros(B, self.energy.hidden_dim, x_mv.shape[-1], device=x_mv.device)
        h_free = self.engine.free_phase(h_init)
        pred_mv = geometric_product(h_free, self.W_out, self.g)
        if self.task_type == 'invariant':
            return scalar_part(pred_mv)
        else:
            return pred_mv[..., 1:4]

def rotate_points(X, R):
    return torch.einsum('ij,bnj->bni', R.to(X.device), X)

def rotate_vector(v, R):
    if v.ndim == 3:
        return torch.einsum('ij,bnj->bni', R.to(v.device), v)
    return torch.einsum('ij,bj->bi', R.to(v.device), v)

# 3. Training and Evaluation

def train_and_eval(task='volume'):
    print(f"--- Task: {task} ---")
    if task == 'volume':
        X, y = generate_convex_hull_data(1200)
        X_train, y_train = X[:1000], y[:1000]
        X_test, y_test = X[1000:], y[1000:]
        task_type, in_dim, out_dim = 'invariant', 20, 1
    elif task == 'force':
        X, q, F_gt = generate_force_field_data(1200)
        X_in = torch.cat([X, q], dim=-1)
        X_train, y_train = X_in[:1000], F_gt[:1000]
        X_test, y_test = X_in[1000:], F_gt[1000:]
        task_type, in_dim, out_dim = 'equivariant', 10, 1
    elif task == 'zn':
        X, y = generate_zn_data(1200)
        X_train, y_train = X[:1000], y[:1000]
        X_test, y_test = X[1000:], y[1000:]
        task_type, in_dim, out_dim = 'invariant', 12, 1
    elif task == 'reversal':
        X, y = generate_time_reversal_data(1200)
        X_train, y_train = X[:1000], y[:1000]
        X_test, y_test = X[1000:], y[1000:]
        task_type, in_dim, out_dim = 'invariant', 5, 1

    signatures = {
        "Cl(3,0)": torch.tensor([1.0, 1.0, 1.0]),
        "CGA Cl(4,1)": torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0]),
        "Cl(1,3)": torch.tensor([1.0, -1.0, -1.0, -1.0]),
    }

    results = {}
    for name, g in signatures.items():
        print(f"Model: {name}")
        sig = CliffordSignature(g) if len(g) < 5 and len(g) != 4 else None
        model = CliffordEPModel(in_dim, 32, out_dim, g, task_type=task_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Pre-embed data
        if task == 'volume' or task == 'zn' or task == 'reversal':
            X_train_mv = embed_vector(X_train, sig, g)
            X_test_mv = embed_vector(X_test, sig, g)
        elif task == 'force':
            X_train_v = embed_vector(X_train[..., :3], sig, g)
            X_train_s = torch.zeros_like(X_train_v); X_train_s[..., 0] = X_train[..., 3]
            X_train_mv = X_train_v + X_train_s
            X_test_v = embed_vector(X_test[..., :3], sig, g)
            X_test_s = torch.zeros_like(X_test_v); X_test_s[..., 0] = X_test[..., 3]
            X_test_mv = X_test_v + X_test_s

        for epoch in range(10):
            preds = model(X_train_mv)
            loss = F.mse_loss(preds.view(y_train.shape), y_train)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if (epoch+1) % 5 == 0: print(f"  Epoch {epoch+1}, Loss: {loss.item():.6f}")

        with torch.no_grad():
            preds_test = model(X_test_mv)
            test_loss = F.mse_loss(preds_test.view(y_test.shape), y_test).item()
            theta = torch.tensor(np.pi / 2, dtype=torch.float32); c, s = torch.cos(theta), torch.sin(theta)
            R = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=torch.float32)
            if task == 'volume' or task == 'zn':
                R_zn = torch.tensor([[c, -s], [s, c]], dtype=torch.float32) if task == 'zn' else R
                X_rot = rotate_points(X_test, R_zn)
                X_rot_mv = embed_vector(X_rot, sig, g)
                preds_rot = model(X_rot_mv)
                violation = torch.norm(preds_rot - preds_test) / (torch.norm(preds_test) + 1e-6)
            elif task == 'force':
                X_pos_rot = rotate_points(X_test[..., :3], R)
                X_rot_in = torch.cat([X_pos_rot, X_test[..., 3:]], dim=-1)
                X_rot_v = embed_vector(X_rot_in[..., :3], sig, g)
                X_rot_s = torch.zeros_like(X_rot_v); X_rot_s[..., 0] = X_rot_in[..., 3]
                X_rot_mv = X_rot_v + X_rot_s
                preds_rot = model(X_rot_mv)
                preds_test_rot = rotate_vector(preds_test, R)
                violation = torch.norm(preds_rot - preds_test_rot) / (torch.norm(preds_test_rot) + 1e-6)
            else: # reversal (T-symmetry test)
                X_rev = X_test.clone(); X_rev[..., 0] *= -1 # reverse time
                X_rev_mv = embed_vector(X_rev, sig, g)
                preds_rev = model(X_rev_mv)
                violation = torch.norm(preds_rev - preds_test) / (torch.norm(preds_test) + 1e-6)

            print(f"  Test Loss: {test_loss:.6f}, Equiv Violation: {violation.item():.6f}")
            results[name] = {"loss": test_loss, "equiv": violation.item()}
    return results

if __name__ == "__main__":
    v_results = train_and_eval('volume')
    f_results = train_and_eval('force')
    zn_results = train_and_eval('zn')
    rev_results = train_and_eval('reversal')
    with open("results/pg2_results.json", "w") as f:
        json.dump({"volume": v_results, "force": f_results, "zn": zn_results, "reversal": rev_results}, f)
