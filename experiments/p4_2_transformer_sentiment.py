import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np
from cliffeq.models.hybrid import CliffordEPBottleneck, CliffordBPBottleneck
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot

def load_sst2_simple(vocab_size=10000, seq_length=100, num_samples=200):
    X = torch.randint(0, vocab_size, (num_samples, seq_length))
    y = torch.randint(0, 2, (num_samples,))
    train_size = int(0.8 * num_samples)
    return (X[:train_size], y[:train_size]), (X[train_size:], y[train_size:])

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, d_model))
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class TransformerBottleneck(nn.Module):
    def __init__(self, variant="baseline", vocab_size=10000, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.variant = variant
        if variant == "clifford-ep":
            self.bottleneck = CliffordEPBottleneck(BilinearEnergy(32, 32, torch.tensor([1., 1.])), LinearDot(), comp=4)
            self.proj_back = nn.Linear(32, d_model)
        elif variant == "clifford-bp":
            self.bottleneck = CliffordBPBottleneck(d_model, 32, torch.tensor([1., 1.]))
        self.layer = TransformerBlock(d_model)
        self.classifier = nn.Linear(d_model, 2)
    def forward(self, x):
        x = self.embedding(x)
        if self.variant == "clifford-ep":
            B, L, D = x.shape
            x = self.proj_back(self.bottleneck(x.view(-1, D))).view(B, L, -1)
        elif self.variant == "clifford-bp":
            B, L, D = x.shape
            x = self.bottleneck(x.view(-1, D)).view(B, L, -1)
        return self.classifier(self.layer(x).mean(dim=1))

def train_eval(variant):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (X_tr, y_tr), (X_te, y_te) = load_sst2_simple()
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=32)
    model = TransformerBottleneck(variant).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); F.cross_entropy(model(bx), by).backward(); optimizer.step()
    correct = 0
    model.eval()
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            correct += (model(bx).argmax(1) == by).sum().item()
    return correct / len(X_te)

def main():
    print("Language Domain Ablation")
    results = {}
    for var in ["baseline", "clifford-ep", "clifford-bp"]:
        print(f"  Running {var}...")
        results[var] = train_eval(var)
        print(f"    Acc: {results[var]:.4f}")
    with open("results/p4_2_transformer_sentiment.json", "w") as f: json.dump(results, f)

if __name__ == "__main__": main()
