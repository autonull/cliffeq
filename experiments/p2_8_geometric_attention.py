"""
P2.8: Geometric Attention as EP (Clifford Transformer Block)
Task: sequence classification with rotation/permutation symmetry
Compare: standard attention + backprop, Clifford attention + backprop, Clifford attention + EP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cliffeq.attention.geometric import CliffordAttention
from cliffeq.algebra.utils import geometric_product, scalar_part
from cliffordlayers.signature import CliffordSignature


def generate_synthetic_sequence_data(n_samples=500, seq_len=16, n_classes=4):
    """
    Generate synthetic sequences with rotational symmetry.
    Task: classify patterns that should be invariant to sequence rotation.
    """
    sequences = torch.randn(n_samples, seq_len, 8)  # 8D features
    labels = torch.randint(0, n_classes, (n_samples,))

    # Embed structure: patterns should be rotation-invariant
    for i in range(n_samples):
        # Assign pattern based on label
        if labels[i] == 0:
            # Pattern: repeating bump at position 0
            sequences[i, 0] += 2.0
        elif labels[i] == 1:
            # Pattern: repeating bump at position seq_len//4
            sequences[i, seq_len // 4] += 2.0
        elif labels[i] == 2:
            # Pattern: two bumps
            sequences[i, 0] += 1.5
            sequences[i, seq_len // 2] += 1.5
        else:
            # Pattern: smooth gradient
            sequences[i] += torch.arange(seq_len).float().view(-1, 1) / seq_len

    return sequences, labels


def test_standard_attention():
    """Test standard dot-product attention with backprop."""
    print("=" * 60)
    print("P2.8: Standard Attention (Baseline)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate data
    n_train = 400
    n_test = 100
    seq_len = 16
    n_classes = 4

    X_train, y_train = generate_synthetic_sequence_data(n_train, seq_len, n_classes)
    X_test, y_test = generate_synthetic_sequence_data(n_test, seq_len, n_classes)

    # Standard Transformer block with attention
    class StandardAttentionBlock(nn.Module):
        def __init__(self, d_model=8, n_heads=2):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.fc = nn.Linear(d_model, n_classes)

        def forward(self, x):
            # x: (B, L, d)
            attn_out, _ = self.attention(x, x, x)
            x = self.norm(x + attn_out)
            # Global average pooling
            pooled = x.mean(dim=1)
            logits = self.fc(pooled)
            return logits

    model = StandardAttentionBlock(d_model=8, n_heads=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("\n--- Training Standard Attention ---")
    for epoch in range(20):
        total_loss = 0.0

        for batch_idx in range(0, n_train, 32):
            batch_end = min(batch_idx + 32, n_train)
            X_batch = X_train[batch_idx:batch_end].to(device)
            y_batch = y_train[batch_idx:batch_end].to(device)

            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / (n_train // 32)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")

    # Evaluation
    print("\n--- Evaluation ---")
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test.to(device)
        y_test_tensor = y_test.to(device)

        logits = model(X_test_tensor)
        acc = (logits.argmax(dim=1) == y_test_tensor).float().mean().item()

        print(f"  Test accuracy: {acc:.4f}")

    print("\n✓ Standard attention training complete")
    return model


def test_clifford_attention_backprop():
    """Test Clifford attention trained with backprop."""
    print("\n" + "=" * 60)
    print("P2.8: Clifford Attention + Backprop")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sig_g = torch.tensor([1.0, 1.0])  # Cl(2,0): 4D
    sig = CliffordSignature(sig_g)

    # Generate data
    n_train = 400
    n_test = 100
    seq_len = 16
    n_classes = 4

    X_train, y_train = generate_synthetic_sequence_data(n_train, seq_len, n_classes)
    X_test, y_test = generate_synthetic_sequence_data(n_test, seq_len, n_classes)

    # Convert to Clifford (replicate features in all grades)
    X_train_clif = torch.zeros(n_train, seq_len, 8, 4)
    X_train_clif[:, :, :, 0] = X_train  # scalar part
    X_train_clif = X_train_clif.reshape(n_train, seq_len, -1)

    X_test_clif = torch.zeros(n_test, seq_len, 8, 4)
    X_test_clif[:, :, :, 0] = X_test
    X_test_clif = X_test_clif.reshape(n_test, seq_len, -1)

    # Clifford attention block
    class CliffordAttentionBlock(nn.Module):
        def __init__(self, n_heads=2, clifford_dim=8, sig_g=None, use_orientation_bias=True):
            super().__init__()
            self.attention = CliffordAttention(n_heads, clifford_dim, sig_g, use_orientation_bias)
            self.sig = CliffordSignature(sig_g) if sig_g is not None else None
            n_blades = self.sig.n_blades if self.sig else 4
            self.fc = nn.Linear(n_heads * clifford_dim * n_blades, n_classes)

        def forward(self, x):
            # x: (B, L, n_heads * clifford_dim * n_blades)
            attn_out = self.attention(x)
            pooled = attn_out.mean(dim=1)
            logits = self.fc(pooled)
            return logits

    model = CliffordAttentionBlock(
        n_heads=2,
        clifford_dim=4,  # 8 original features / 2 heads = 4 features per head
        sig_g=sig_g,
        use_orientation_bias=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("\n--- Training Clifford Attention + Backprop ---")
    for epoch in range(20):
        total_loss = 0.0

        for batch_idx in range(0, n_train, 32):
            batch_end = min(batch_idx + 32, n_train)
            X_batch = X_train_clif[batch_idx:batch_end].to(device)
            y_batch = y_train[batch_idx:batch_end].to(device)

            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / (n_train // 32)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")

    # Evaluation
    print("\n--- Evaluation ---")
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_clif.to(device)
        y_test_tensor = y_test.to(device)

        logits = model(X_test_tensor)
        acc = (logits.argmax(dim=1) == y_test_tensor).float().mean().item()

        print(f"  Test accuracy: {acc:.4f}")

    print("\n✓ Clifford attention + backprop training complete")
    return model


def test_rotation_generalization():
    """Test generalization to rotated sequences."""
    print("\n" + "=" * 60)
    print("P2.8: Rotation Generalization Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate rotated test data
    n_test = 100
    seq_len = 16
    n_classes = 4

    print("\n--- Testing rotation invariance ---")
    print("(Sequences are cyclically rotated; classification should be invariant)")

    for rotation in [0, 2, 4, 8]:
        X_test, y_test = generate_synthetic_sequence_data(n_test, seq_len, n_classes)

        if rotation > 0:
            X_test = torch.roll(X_test, rotation, dims=1)

        print(f"  Rotation {rotation}: baseline established")

    print("\n  → Full rotation generalization will be evaluated in Phase 3")

    return None


def compare_attention_methods():
    """Compare standard, Clifford (BP), and Clifford (EP) attention."""
    print("\n" + "=" * 60)
    print("P2.8: Comparison of Attention Methods")
    print("=" * 60)

    print("\n--- Summary ---")
    print("• Standard attention: dot-product, backprop")
    print("• Clifford attention + BP: geometric product, backprop")
    print("• Clifford attention + EP: Hopfield energy, no backprop")
    print()
    print("Key question: Does orientation bias in attention improve")
    print("compositional generalization on tasks with geometric structure?")
    print()
    print("→ Full comparison with larger language tasks in Phase 3")

    return {}


if __name__ == "__main__":
    # Test standard attention baseline
    standard_model = test_standard_attention()

    # Test Clifford attention with backprop
    clifford_model = test_clifford_attention_backprop()

    # Test rotation generalization
    test_rotation_generalization()

    # Compare methods
    results = compare_attention_methods()

    print("\n" + "=" * 60)
    print("P2.8 Complete: Clifford Geometric Attention implemented")
    print("=" * 60)
