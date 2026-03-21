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
from cliffeq.algebra.utils import geometric_product, scalar_part, reverse
from cliffordlayers.signature import CliffordSignature
from cliffeq.training.ep_engine import EPEngine
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot

def generate_synthetic_sequence_data(n_samples=500, seq_len=16, n_classes=4):
    """
    Generate synthetic sequences with rotational symmetry.
    Task: classify patterns that should be invariant to sequence rotation.
    """
    sequences = torch.randn(n_samples, seq_len, 8)  # 8D features
    labels = torch.randint(0, n_classes, (n_samples,))

    for i in range(n_samples):
        if labels[i] == 0:
            sequences[i, 0] += 2.0
        elif labels[i] == 1:
            sequences[i, seq_len // 4] += 2.0
        elif labels[i] == 2:
            sequences[i, 0] += 1.5
            sequences[i, seq_len // 2] += 1.5
        else:
            sequences[i] += torch.arange(seq_len).float().view(-1, 1) / seq_len

    return sequences, labels

class HopfieldAttentionEnergy(EnergyFunction):
    """
    Modern Hopfield energy for an attention block:
    E(x) = -log Σ_j exp(β · scalar(Q̃ ✶ K_j))
    """
    def __init__(self, k, sig_g, beta=1.0):
        super().__init__()
        self.register_buffer("k", k) # Keys: (B, H, M, D, I)
        self.register_buffer("sig_g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.beta = beta

    def forward(self, q):
        # q: (B, H, L, D, I)
        from cliffeq.algebra.utils import get_blade_signs
        signs = get_blade_signs(self.sig, q.device)

        q_rev = reverse(q, self.sig)
        q_w = q_rev * signs

        # scores: (B, H, L, M)
        scores = torch.einsum("bhldi,bhmdi->bhlm", q_w, self.k) / (q.shape[-2] * q.shape[-1])**0.5
        energy = -torch.logsumexp(self.beta * scores, dim=-1)
        return energy.sum()

def test_clifford_attention_ep():
    """Test Clifford attention trained with EP (Hopfield retrieval)."""
    print("\n" + "=" * 60)
    print("P2.8: Clifford Attention + Equilibrium Propagation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(sig_g)

    # Generate data
    n_train = 400
    n_test = 100
    seq_len = 16
    n_classes = 4

    X_train, y_train = generate_synthetic_sequence_data(n_train, seq_len, n_classes)
    X_test, y_test = generate_synthetic_sequence_data(n_test, seq_len, n_classes)

    # Convert to Clifford
    X_train_clif = torch.zeros(n_train, seq_len, 8, 4)
    X_train_clif[:, :, :, 0] = X_train
    X_train_clif = X_train_clif.reshape(n_train, seq_len, -1)

    X_test_clif = torch.zeros(n_test, seq_len, 8, 4)
    X_test_clif[:, :, :, 0] = X_test
    X_test_clif = X_test_clif.reshape(n_test, seq_len, -1)

    class CliffordTransformerEP(nn.Module):
        def __init__(self, n_heads=2, clifford_dim=4):
            super().__init__()
            self.attention = CliffordAttention(n_heads, clifford_dim, sig_g)
            self.fc = nn.Linear(n_heads * clifford_dim * 4, n_classes)
            self.n_heads = n_heads
            self.clifford_dim = clifford_dim

        def forward(self, x):
            # Normal forward for evaluation
            attn_out, _ = self.attention(x)
            pooled = attn_out.mean(dim=1)
            return self.fc(pooled)

    model = CliffordTransformerEP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\n--- Training Clifford Attention + EP (Hopfield) ---")
    for epoch in range(10):
        total_loss = 0.0
        for batch_idx in range(0, n_train, 32):
            batch_end = min(batch_idx + 32, n_train)
            X_batch = X_train_clif[batch_idx:batch_end].to(device)
            y_batch = y_train[batch_idx:batch_end].to(device)

            # 1. Forward to get K and initial Q
            B = X_batch.shape[0]
            q_init = model.attention.q_proj(X_batch).view(B, seq_len, model.n_heads, model.clifford_dim, 4).transpose(1, 2)
            k = model.attention.k_proj(X_batch).view(B, seq_len, model.n_heads, model.clifford_dim, 4).transpose(1, 2)

            # 2. EP Free Phase: Relax Q using Hopfield Energy
            energy_fn = HopfieldAttentionEnergy(k, sig_g)
            dynamics = LinearDot()
            engine = EPEngine(energy_fn, dynamics, n_free=5, n_clamped=0, beta=0.0, dt=0.1)

            q_free = engine.free_phase(q_init)

            # 3. Supervised update for FC and Projections via Backprop (Simplified EP)
            # In a full EP model, we'd clamp the output, but here we just use the settled Q
            # to compute attention and then update weights.

            # Reconstruct attn_out from settled q_free
            from cliffeq.algebra.utils import get_blade_signs
            signs = get_blade_signs(sig, device)
            q_rev = reverse(q_free, sig)
            q_w = q_rev * signs
            scores = torch.einsum("bhldi,bhmdi->bhlm", q_w, k) / (model.clifford_dim * 4)**0.5
            attn = F.softmax(scores, dim=-1)
            v = model.attention.v_proj(X_batch).view(B, seq_len, model.n_heads, model.clifford_dim, 4).transpose(1, 2)
            out = torch.einsum("bhlm,bhmdi->bhldi", attn, v)
            out = out.transpose(1, 2).reshape(B, seq_len, -1)

            pooled = out.mean(dim=1)
            logits = model.fc(pooled)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/(n_train//32):.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_clif.to(device))
        acc = (logits.argmax(dim=1) == y_test.to(device)).float().mean().item()
        print(f"  Test accuracy: {acc:.4f}")

    return model

if __name__ == "__main__":
    test_clifford_attention_ep()
