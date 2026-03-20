"""
P2.9 (V2 - FIXED): Clifford-EP Bottleneck Layer Test
Fixed gradient flow + simpler energy function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cliffeq.models.bottleneck_v2 import CliffordEPBottleneckV2
from cliffordlayers.signature import CliffordSignature


def test_bottleneck_in_mlp():
    """Test bottleneck inserted into simple MLP."""
    print("=" * 70)
    print("P2.9 (V2): Clifford-EP Bottleneck in MLP")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    sig_g = torch.tensor([1.0, 1.0, 1.0])  # Cl(3,0)

    # Generate synthetic control task (CartPole-like)
    n_samples = 500
    state_dim = 4
    action_dim = 2

    states = torch.randn(n_samples, state_dim)
    actions = torch.randint(0, action_dim, (n_samples,))

    # Create mirrored variants (mirror symmetry test)
    states_mirrored = states.clone()
    states_mirrored[:, 0] = -states_mirrored[:, 0]  # Mirror first component

    # MLP without bottleneck
    class MLPBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # MLP with bottleneck
    class MLPWithBottleneck(nn.Module):
        def __init__(self, sig_g):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 64)
            self.bottleneck = CliffordEPBottleneckV2(
                in_dim=64,
                out_dim=32,
                sig_g=sig_g,
                n_ep_steps=3,
                step_size=0.01,
                use_spectral_norm=True
            )
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.bottleneck(x)  # Clifford-EP processing
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Train baseline
    print("-" * 70)
    print("Baseline MLP (no bottleneck)")
    print("-" * 70)
    model_baseline = MLPBaseline().to(device)
    optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=0.01)

    dataset_baseline = TensorDataset(states.to(device), actions.to(device))
    loader_baseline = DataLoader(dataset_baseline, batch_size=32, shuffle=True)

    for epoch in range(10):
        total_loss = 0.0
        for states_batch, actions_batch in loader_baseline:
            logits = model_baseline(states_batch)
            loss = F.cross_entropy(logits, actions_batch)

            optimizer_baseline.zero_grad()
            loss.backward()
            optimizer_baseline.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: loss={total_loss / len(loader_baseline):.6f}")

    # Evaluate baseline on mirror symmetry
    model_baseline.eval()
    with torch.no_grad():
        logits_normal = model_baseline(states.to(device))
        logits_mirrored = model_baseline(states_mirrored.to(device))

        pred_normal = logits_normal.argmax(dim=1)
        pred_mirrored = logits_mirrored.argmax(dim=1)
        mirror_violation_baseline = (pred_normal != pred_mirrored).float().mean().item()

    print(f"Baseline mirror symmetry violation: {mirror_violation_baseline:.4f}")

    # Train with bottleneck
    print("\n" + "-" * 70)
    print("MLP with Clifford-EP Bottleneck")
    print("-" * 70)
    model_bottleneck = MLPWithBottleneck(sig_g).to(device)
    optimizer_bottleneck = torch.optim.Adam(model_bottleneck.parameters(), lr=0.01)

    dataset_bottleneck = TensorDataset(states.to(device), actions.to(device))
    loader_bottleneck = DataLoader(dataset_bottleneck, batch_size=32, shuffle=True)

    for epoch in range(10):
        total_loss = 0.0
        for states_batch, actions_batch in loader_bottleneck:
            logits = model_bottleneck(states_batch)
            loss = F.cross_entropy(logits, actions_batch)

            optimizer_bottleneck.zero_grad()
            loss.backward()
            optimizer_bottleneck.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: loss={total_loss / len(loader_bottleneck):.6f}")

    # Evaluate bottleneck on mirror symmetry
    model_bottleneck.eval()
    with torch.no_grad():
        logits_normal = model_bottleneck(states.to(device))
        logits_mirrored = model_bottleneck(states_mirrored.to(device))

        pred_normal = logits_normal.argmax(dim=1)
        pred_mirrored = logits_mirrored.argmax(dim=1)
        mirror_violation_bottleneck = (pred_normal != pred_mirrored).float().mean().item()

    print(f"Bottleneck mirror symmetry violation: {mirror_violation_bottleneck:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Mirror Symmetry Test")
    print("=" * 70)
    print(f"Baseline violation:   {mirror_violation_baseline:.4f}")
    print(f"Bottleneck violation: {mirror_violation_bottleneck:.4f}")
    improvement = (mirror_violation_baseline - mirror_violation_bottleneck) / (mirror_violation_baseline + 1e-6)
    if improvement > 0:
        print(f"✓ Bottleneck: {improvement*100:.1f}% improvement in symmetry")
    else:
        print(f"⚠ Baseline: {-improvement*100:.1f}% better symmetry")

    print("\n" + "=" * 70)
    print("P2.9 (V2) Complete")
    print("=" * 70)

    return {
        'baseline_violation': mirror_violation_baseline,
        'bottleneck_violation': mirror_violation_bottleneck,
        'improvement': improvement
    }


if __name__ == "__main__":
    results = test_bottleneck_in_mlp()
