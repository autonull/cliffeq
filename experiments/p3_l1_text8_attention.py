import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffeq.attention.geometric import CliffordAttention
from cliffeq.algebra.utils import embed_vector
import time
import os
import json
import urllib.request
import zipfile
import numpy as np

def download_text8():
    url = "http://mattmahoney.net/dc/text8.zip"
    if not os.path.exists("/tmp/text8"):
        print("Downloading text8...")
        urllib.request.urlretrieve(url, "/tmp/text8.zip")
        with zipfile.ZipFile("/tmp/text8.zip", "r") as f:
            f.extractall("/tmp")
    with open("/tmp/text8", "r") as f:
        return f.read()

def prepare_data(text, seq_len=64, n_samples=2000):
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    data = []
    for i in range(0, min(len(text)-seq_len-1, n_samples * seq_len), seq_len):
        chunk = text[i:i+seq_len+1]
        data.append([char_to_idx[c] for c in chunk])

    data = torch.tensor(data)
    X = data[:, :-1]
    y = data[:, 1:]
    return X, y, vocab_size

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, clifford_dim, sig_g):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = CliffordAttention(n_heads, clifford_dim, sig_g)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, mask=None):
        B, L = x.shape
        h = self.embedding(x)
        # MultiheadAttention-like signature
        h, _ = self.attention(h, h, h, attn_mask=mask)
        logits = self.fc(h)
        return logits

def run_text8_attention():
    print("P2.8: text8 Character Language Model with Clifford Attention")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = download_text8()
    X, y, vocab_size = prepare_data(text, seq_len=64, n_samples=1000)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    sig_g = torch.tensor([1.0, 1.0])
    # Cl(2,0) -> 4 blades
    # n_heads * clifford_dim * 4 = d_model
    n_heads = 4
    clifford_dim = 4
    d_model = n_heads * clifford_dim * 4 # 64

    model = SimpleTransformer(vocab_size, d_model, n_heads, clifford_dim, sig_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Causal mask
    mask = torch.triu(torch.ones(64, 64, device=device), diagonal=1) * -1e9

    print("Training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), 32):
            batch_X = X_train[i:i+32].to(device)
            batch_y = y_train[i:i+32].to(device)

            logits = model(batch_X, mask=mask)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}, Loss: {total_loss / (len(X_train)//32 + 1):.4f}")

    # Eval
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(X_test), 32):
            batch_X = X_test[i:i+32].to(device)
            batch_y = y_test[i:i+32].to(device)
            logits = model(batch_X, mask=mask)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
            total_loss += loss.item()

    avg_test_loss = total_loss / (len(X_test)//32 + 1)
    bpc = avg_test_loss / np.log(2)
    print(f"Final Test BPC: {bpc:.4f}")
    return {"bpc": bpc}

if __name__ == "__main__":
    results = run_text8_attention()
    os.makedirs("results", exist_ok=True)
    with open("results/p3_l1_text8_attention.json", "w") as f:
        json.dump(results, f)
