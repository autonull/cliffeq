"""
PL3: Energy-Based Clifford Sequence Model (JEPA-style)
Task: Predict future latent representations using Clifford-EP.
Domain: Language/Sequences - Speculative test for semantic latents.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import copy

from cliffeq.models.flat import EPModel
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot
from cliffeq.algebra.utils import embed_vector, scalar_part
from cliffordlayers.signature import CliffordSignature

class SyntheticSequenceDataset(Dataset):
    def __init__(self, n_samples=300, seq_len=16, vocab_size=20):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Generate correlated sequences
        # Rule: next sequence is a "transformed" version of context
        self.data = []
        for _ in range(n_samples):
            # Context sequence
            base = np.random.randint(0, vocab_size, seq_len)
            # Target sequence (e.g., shifted or slightly modified)
            target = np.roll(base, 1)
            self.data.append((base, target))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ctx, tgt = self.data[idx]
        return torch.tensor(ctx).long(), torch.tensor(tgt).long()

class CliffordEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, sig_g):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sig = CliffordSignature(sig_g)
        self.n_blades = self.sig.n_blades

        # Simple GRU or Transformer-like encoder
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, self.n_blades) # Project to a single multivector

    def forward(self, x):
        # x: (B, L)
        emb = self.embedding(x)
        _, h = self.gru(emb) # h: (1, B, d)
        h = h.squeeze(0)
        mv = self.fc(h) # (B, I)
        return mv.unsqueeze(1) # (B, 1, I)

class CliffordJEPAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, sig_g):
        super().__init__()
        self.sig = CliffordSignature(sig_g)

        # Context Encoder
        self.context_encoder = CliffordEncoder(vocab_size, embed_dim, sig_g)

        # Target Encoder (EMA)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor (Clifford-EP)
        self.predictor_energy = BilinearEnergy(
            in_nodes=1,
            hidden_nodes=1,
            sig_g=sig_g,
            use_spectral_norm=True
        )
        self.predictor = EPModel(
            energy_fn=self.predictor_energy,
            dynamics_rule=LinearDot(),
            n_free=5,
            n_clamped=0,
            beta=0.0,
            dt=0.1
        )

    def update_target_encoder(self, momentum=0.99):
        with torch.no_grad():
            for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                p_tgt.data = p_tgt.data * momentum + p_ctx.data * (1 - momentum)

    def forward(self, ctx, tgt_seq):
        # ctx_latent: (B, 1, I)
        ctx_latent = self.context_encoder(ctx)

        # Predict future latent
        # Set context as input to BilinearEnergy
        self.predictor_energy.set_input(ctx_latent)

        # Differentiable unroll to ensure gradients flow for context_encoder and predictor weights
        # We use torch.enable_grad() to ensure this works even inside torch.no_grad()
        with torch.enable_grad():
            h = torch.zeros_like(ctx_latent).requires_grad_(True)
            for _ in range(5):
                E = self.predictor_energy(h).sum()
                grad = torch.autograd.grad(E, h, create_graph=True)[0]
                h = h - 0.1 * grad
        pred_latent = h

        # Real target latent (from Target Encoder)
        with torch.no_grad():
            target_latent = self.target_encoder(tgt_seq)

        return pred_latent, target_latent

def run_pl3():
    print("=" * 80)
    print("PL3: Energy-Based Clifford Sequence Model (JEPA-style)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_g = torch.tensor([1.0, 1.0]) # Cl(2,0): 4D

    vocab_size = 50
    embed_dim = 64
    dataset = SyntheticSequenceDataset(n_samples=400, vocab_size=vocab_size)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CliffordJEPAModel(vocab_size, embed_dim, sig_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training JEPA-style Clifford model...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for ctx, tgt in train_loader:
            ctx, tgt = ctx.to(device), tgt.to(device)

            pred_latent, target_latent = model(ctx, tgt)

            # Loss: MSE in Clifford latent space
            loss = F.mse_loss(pred_latent, target_latent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA Update
            model.update_target_encoder(momentum=0.99)

            total_loss += loss.item()
        print(f"  Epoch {epoch+1}, Latent MSE: {total_loss/len(train_loader):.6f}")

    # Final Eval
    model.eval()
    test_dataset = SyntheticSequenceDataset(n_samples=100, vocab_size=vocab_size)
    test_loader = DataLoader(test_dataset, batch_size=32)

    eval_loss = 0
    with torch.no_grad():
        for ctx, tgt in test_loader:
            ctx, tgt = ctx.to(device), tgt.to(device)
            pred, target = model(ctx, tgt)
            eval_loss += F.mse_loss(pred, target).item()

    final_mse = eval_loss / len(test_loader)
    print(f"\nFinal Test Latent MSE: {final_mse:.6f}")

    results = {"latent_prediction_mse": final_mse}
    os.makedirs("results", exist_ok=True)
    with open("results/pl3_results.json", "w") as f:
        json.dump(results, f)
    print("\n✓ PL3 Complete")

if __name__ == "__main__":
    run_pl3()
