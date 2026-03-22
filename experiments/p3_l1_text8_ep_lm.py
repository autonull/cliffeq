import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import urllib.request
import zipfile
import numpy as np
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot
from cliffeq.training.ep_engine import EPEngine
from cliffeq.algebra.utils import embed_vector, scalar_part, get_blade_signs, geometric_product
from cliffordlayers.signature import CliffordSignature

# 1. Data loading
def download_text8():
    path = "./data/text8"
    if not os.path.exists(path):
        os.makedirs("./data", exist_ok=True)
        url = "http://mattmahoney.net/dc/text8.zip"
        print("Downloading text8...")
        urllib.request.urlretrieve(url, "./data/text8.zip")
        with zipfile.ZipFile("./data/text8.zip", "r") as f:
            f.extractall("./data")
    with open(path, "r") as f:
        return f.read()

def prepare_data(text, seq_len=32, n_samples=1000):
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    data = []
    # Use a larger range for better coverage
    for i in range(0, min(len(text)-seq_len-1, n_samples * seq_len), seq_len):
        chunk = text[i:i+seq_len+1]
        if len(chunk) < seq_len + 1: continue
        data.append([char_to_idx[c] for c in chunk])

    data = torch.tensor(data)
    X = data[:, :-1]
    y = data[:, 1:]
    return X, y, vocab_size

# 2. Causal Clifford Energy
class CausalCliffordEnergy(EnergyFunction):
    def __init__(self, vocab_size, embed_dim, hidden_dim, sig_g):
        super().__init__()
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.n_blades = self.sig.n_blades
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.W_in = nn.Parameter(torch.randn(hidden_dim, embed_dim, self.n_blades) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(hidden_dim, hidden_dim, self.n_blades) * 0.05)

        self.input_mv = None

    def set_input(self, x_indices):
        # x_indices: (B, L)
        B, L = x_indices.shape
        emb = self.embedding(x_indices) # (B, L, D)
        self.input_mv = torch.einsum("bld,hdi->blhi", emb, self.W_in)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, L, H, I)
        B, L, H, I = h.shape
        rho_h = torch.clamp(h, 0, 1)

        # Self energy
        E_self = 0.5 * torch.sum(h**2, dim=(1, 2, 3))

        # Input interaction
        signs = get_blade_signs(self.sig, h.device)
        E_int_in = torch.einsum("blhi,blhi,i->b", rho_h, self.input_mv, signs)

        # Recurrent causal interaction
        E_int_rec = 0
        if L > 1:
            h_curr = rho_h[:, 1:] # (B, L-1, H, I)
            h_prev = rho_h[:, :-1] # (B, L-1, H, I)
            B_sub, L_sub, H_sub, I_sub = h_prev.shape
            h_prev_flat = h_prev.reshape(B_sub * L_sub, H_sub, I_sub)
            rec_term_flat = geometric_product(h_prev_flat, self.W_rec, self.g)
            rec_term = rec_term_flat.reshape(B_sub, L_sub, H_sub, I_sub)

            E_int_rec = torch.einsum("blhi,blhi,i->b", h_curr, rec_term, signs)

        return E_self - E_int_in - E_int_rec

class CliffordEPLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, sig_g):
        super().__init__()
        self.energy = CausalCliffordEnergy(vocab_size, embed_dim, hidden_dim, sig_g)
        self.rule = LinearDot()
        self.engine = EPEngine(self.energy, self.rule, n_free=5, n_clamped=5, beta=0.1, dt=0.1)
        self.W_out = nn.Parameter(torch.randn(vocab_size, hidden_dim, self.energy.n_blades) * 0.1)

    def forward(self, x):
        B, L = x.shape
        self.energy.set_input(x)
        h_init = torch.zeros(B, L, self.energy.hidden_dim, self.energy.n_blades, device=x.device)
        h_free = self.engine.free_phase(h_init)

        h_free_flat = h_free.reshape(B * L, self.energy.hidden_dim, self.energy.n_blades)
        logits_mv_flat = geometric_product(h_free_flat, self.W_out, self.energy.g) # (B*L, V, I)
        logits_mv = logits_mv_flat.reshape(B, L, -1, self.energy.n_blades)
        logits = scalar_part(logits_mv) # (B, L, V)
        return logits

def run_pl1():
    print("PL1: Clifford Token Embeddings + EP Language Model on text8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = download_text8()
    seq_len = 32
    n_samples = 1000
    batch_size = 32
    n_epochs = 10

    X, y, vocab_size = prepare_data(text, seq_len=seq_len, n_samples=n_samples)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    print(f"Dataset: {len(X_train)} train samples, {len(X_test)} test samples")

    sig_g = torch.tensor([1.0, 1.0, 1.0]) # Cl(3,0)
    model = CliffordEPLM(vocab_size, embed_dim=64, hidden_dim=64, sig_g=sig_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    print("Training...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            if len(batch_X) == 0: continue

            logits = model(batch_X)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches if n_batches > 0 else 0
        print(f"  Epoch {epoch+1}/{n_epochs}, Avg Loss: {avg_train_loss:.4f}")

    # Eval
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size].to(device)
            if len(batch_X) == 0: continue
            logits = model(batch_X)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    bpc = avg_loss / np.log(2)
    print(f"Final Test BPC: {bpc:.4f}")

    results = {"bpc": bpc, "epochs": n_epochs, "n_samples": n_samples, "seq_len": seq_len}
    os.makedirs("results", exist_ok=True)
    with open("results/pl1_results.json", "w") as f:
        json.dump(results, f)
    print("PL1 Complete")

if __name__ == "__main__":
    run_pl1()
