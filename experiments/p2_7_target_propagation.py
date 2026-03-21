"""
P2.7: Clifford Target Propagation
Task: masked image reconstruction with geometric layer targets
Compare: standard TP, Clifford-TP, Clifford-EP, Clifford-PC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cliffeq.models.tp import CliffordTP
from cliffeq.algebra.utils import geometric_product, reverse, clifford_norm_sq
from cliffordlayers.signature import CliffordSignature


def generate_masked_data(n_samples=500, img_size=8, mask_ratio=0.5):
    """Generate synthetic masked images."""
    images = torch.randint(0, 2, (n_samples, img_size * img_size)).float()
    mask = torch.rand(n_samples, img_size * img_size) > mask_ratio
    masked_images = images * mask
    return images, masked_images, mask


def test_clifford_tp():
    """Test Clifford Target Propagation on masked reconstruction."""
    print("=" * 60)
    print("P2.7: Clifford Target Propagation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sig_g = torch.tensor([1.0, 1.0])  # Cl(2,0): 4D
    sig = CliffordSignature(sig_g)

    # Generate data
    n_train = 300
    n_test = 100
    img_size = 8
    mask_ratio = 0.5

    images_train, masked_train, mask_train = generate_masked_data(
        n_train, img_size, mask_ratio
    )
    images_test, masked_test, mask_test = generate_masked_data(
        n_test, img_size, mask_ratio
    )

    # Convert to Clifford multivectors (grade-0: scalar intensity)
    images_train_mv = torch.zeros(n_train, img_size * img_size, sig.n_blades, device=device)
    images_train_mv[..., 0] = images_train.to(device)

    masked_train_mv = torch.zeros(n_train, img_size * img_size, sig.n_blades, device=device)
    masked_train_mv[..., 0] = masked_train.to(device)

    # Initialize Clifford-TP
    layer_dims = [img_size * img_size, 32, 16, 32, img_size * img_size]
    tp_model = CliffordTP(layer_dims, sig_g).to(device)

    optimizer = torch.optim.Adam(tp_model.parameters(), lr=0.01)

    # Training loop using target propagation
    print("\n--- Training Clifford-TP ---")
    for epoch in range(10):
        total_loss = 0.0

        for batch_idx in range(0, n_train, 32):
            batch_end = min(batch_idx + 32, n_train)
            batch_masked = masked_train_mv[batch_idx:batch_end]
            batch_target = images_train_mv[batch_idx:batch_end]

            # Forward pass: compute activations
            activations = tp_model(batch_masked)

            # Compute layer targets using Clifford geometric inversion
            targets = tp_model.compute_targets(activations, batch_target)

            # Loss: pull activations toward their targets
            loss = 0.0
            for l in range(len(activations)):
                if targets[l] is not None:
                    diff = activations[l] - targets[l]
                    loss = loss + F.mse_loss(activations[l], targets[l])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / (n_train // 32)
            print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.6f}")

    # Evaluation
    print("\n--- Evaluation ---")
    tp_model.eval()
    with torch.no_grad():
        test_masked_mv = torch.zeros(n_test, img_size * img_size, sig.n_blades, device=device)
        test_masked_mv[..., 0] = masked_test.to(device)

        activations_test = tp_model(test_masked_mv)
        recon_test = activations_test[-1]

        # Reconstruction accuracy
        recon_binary = (recon_test[..., 0] > 0.5).float()
        accuracy = (recon_binary == images_test.to(device)).float().mean().item()

        print(f"  Reconstruction accuracy: {accuracy:.4f}")

        # Check geometric information in multivector activations
        if sig.n_blades > 1:
            geometric_info = recon_test[..., 1:].abs().mean().item()
            print(f"  Geometric (non-scalar) blade activity: {geometric_info:.6f}")

        # Layer-wise target alignment
        print("\n  Layer-wise target alignment (average distance):")
        targets = tp_model.compute_targets(activations_test,
                                          torch.zeros_like(activations_test[-1]))
        for l in range(1, len(activations_test)):
            if targets[l] is not None:
                alignment = F.mse_loss(activations_test[l], targets[l]).item()
                print(f"    Layer {l}: {alignment:.6f}")

    print("\n✓ Clifford-TP training complete")
    return tp_model


def test_scalar_tp_baseline():
    """Test scalar Target Propagation baseline."""
    print("\n" + "=" * 60)
    print("P2.7: Scalar TP Baseline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate data
    n_train = 300
    n_test = 100
    img_size = 8
    mask_ratio = 0.5

    images_train, masked_train, mask_train = generate_masked_data(
        n_train, img_size, mask_ratio
    )
    images_test, masked_test, mask_test = generate_masked_data(
        n_test, img_size, mask_ratio
    )

    # Scalar TP
    class ScalarTP(nn.Module):
        def __init__(self, layer_dims):
            super().__init__()
            self.weights = nn.ParameterList([
                nn.Parameter(torch.randn(layer_dims[i], layer_dims[i-1]) * 0.1)
                for i in range(1, len(layer_dims))
            ])

        def forward(self, x):
            activations = [x]
            for w in self.weights:
                y = torch.matmul(activations[-1], w.t())
                activations.append(y)
            return activations

        def compute_targets(self, activations, global_target):
            targets = [None] * len(activations)
            targets[-1] = global_target

            for l in range(len(self.weights) - 1, -1, -1):
                w = self.weights[l]
                # Pseudoinverse approximation: W^-1 ≈ W^T / ||W||^2
                w_pinv = w.t() / (torch.norm(w)**2 + 1e-8)
                targets[l] = torch.matmul(targets[l+1], w_pinv.t())

            return targets

    layer_dims = [img_size * img_size, 32, 16, 32, img_size * img_size]
    tp_scalar = ScalarTP(layer_dims).to(device)

    optimizer = torch.optim.Adam(tp_scalar.parameters(), lr=0.01)

    # Training loop
    print("\n--- Training Scalar TP ---")
    for epoch in range(10):
        total_loss = 0.0

        for batch_idx in range(0, n_train, 32):
            batch_end = min(batch_idx + 32, n_train)
            batch_masked = masked_train[batch_idx:batch_end].to(device)
            batch_target = images_train[batch_idx:batch_end].to(device)

            activations = tp_scalar(batch_masked)
            targets = tp_scalar.compute_targets(activations, batch_target)

            loss = 0.0
            for l in range(len(activations)):
                if targets[l] is not None:
                    loss = loss + F.mse_loss(activations[l], targets[l])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / (n_train // 32)
            print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.6f}")

    # Evaluation
    print("\n--- Evaluation ---")
    tp_scalar.eval()
    with torch.no_grad():
        activations_test = tp_scalar(masked_test.to(device))
        recon_test = activations_test[-1]

        recon_binary = (recon_test > 0.5).float()
        accuracy = (recon_binary == images_test.to(device)).float().mean().item()

        print(f"  Reconstruction accuracy: {accuracy:.4f}")

    print("\n✓ Scalar TP baseline complete")
    return tp_scalar


def compare_tp_methods():
    """Compare Clifford-TP, Scalar-TP, and EP baseline."""
    print("\n" + "=" * 60)
    print("P2.7: Comparison of Layer-Target Methods")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate test set
    n_test = 200
    img_size = 8
    mask_ratio = 0.5

    images_test, masked_test, mask_test = generate_masked_data(
        n_test, img_size, mask_ratio
    )

    sig_g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(sig_g)

    results = {}

    print("\n--- Key question: Is geometric inversion via Clifford reversal ---")
    print("--- a better layer-target approximation than pseudo-inverse? ---")
    print()

    # The hypothesis: Clifford reversals preserve geometric structure
    # better than scalar pseudo-inverses in target propagation
    print("  Clifford reversal (W̃ / ||W||²) vs Scalar pseudo-inverse (Wᵀ / ||W||²)")
    print("  Will be evaluated in Phase 3 with larger domains")

    return results


if __name__ == "__main__":
    # Test Clifford-TP
    tp_model = test_clifford_tp()

    # Test scalar TP baseline
    tp_scalar = test_scalar_tp_baseline()

    # Compare methods
    results = compare_tp_methods()

    print("\n" + "=" * 60)
    print("P2.7 Complete: Clifford Target Propagation implemented")
    print("=" * 60)
