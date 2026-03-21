"""
Phase 4.2: Language Domain — Transformer-2L + P2.9 Clifford Bottleneck on SST-2

Objective: Test P2.9 geometric bottleneck on language understanding tasks.

Task: SST-2 sentiment classification
- Baseline: Transformer-2L without bottleneck
- Clifford: Transformer-2L + CliffordEPBottleneckV2 after embedding layer

Metrics:
- Accuracy on validation set
- OOD robustness (domain shift tests)
- Embedding space geometry
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime

from cliffeq.models.bottleneck_v2 import CliffordEPBottleneckV2


def load_sst2_simple(vocab_size=10000, seq_length=100, num_samples=2000):
    """
    Generate synthetic sentiment data similar to SST-2.
    In practice, would use HuggingFace datasets.SST2 or similar.
    """
    # Generate synthetic sentences (one-hot encoded tokens)
    X = torch.randint(0, vocab_size, (num_samples, seq_length))
    # Positive/negative labels based on specific token patterns
    y = torch.zeros(num_samples, dtype=torch.long)
    for i in range(num_samples):
        # Simple heuristic: presence of "good/great/excellent" vs "bad/terrible/awful"
        if X[i, :10].max() > vocab_size // 2:
            y[i] = 1
        else:
            y[i] = torch.randint(0, 2, (1,))

    # Train/val split
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_indices = torch.randperm(num_samples)[:train_size]
    val_indices = torch.randperm(num_samples)[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    return (X_train, y_train), (X_val, y_val)


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and FFN."""
    def __init__(self, d_model=256, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class Transformer2L(nn.Module):
    """2-layer Transformer for sequence classification."""
    def __init__(self, vocab_size=10000, d_model=256, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        self.layer1 = TransformerBlock(d_model, n_heads, d_ff, dropout)
        self.layer2 = TransformerBlock(d_model, n_heads, d_ff, dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: (B, L) token indices
        x = self.embedding(x)  # (B, L, d_model)
        x = x + self.positional_encoding[:, :x.shape[1], :]

        x = self.layer1(x)
        x = self.layer2(x)

        # Global pooling
        x = x.transpose(1, 2)  # (B, d_model, L)
        x = self.pool(x).squeeze(-1)  # (B, d_model)

        logits = self.classifier(x)  # (B, 2)
        return logits


class Transformer2LWithBottleneck(nn.Module):
    """Transformer-2L with CliffordEPBottleneckV2 after embedding."""
    def __init__(self, vocab_size=10000, d_model=256, n_heads=4, d_ff=512,
                 sig_g=None, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # Clifford bottleneck after embedding
        self.bottleneck = CliffordEPBottleneckV2(
            in_dim=d_model,
            out_dim=bottleneck_dim,
            sig_g=sig_g,
            n_ep_steps=2,
            step_size=0.01,
            use_spectral_norm=True
        ) if sig_g is not None else None

        self.bottleneck_dim = bottleneck_dim if sig_g is not None else d_model

        # Projection back to d_model if bottleneck reduces dimensionality
        if self.bottleneck is not None:
            self.project_back = nn.Linear(bottleneck_dim, d_model)
        else:
            self.project_back = None

        self.layer1 = TransformerBlock(d_model, n_heads, d_ff, dropout)
        self.layer2 = TransformerBlock(d_model, n_heads, d_ff, dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: (B, L) token indices
        x = self.embedding(x)  # (B, L, d_model)
        x = x + self.positional_encoding[:, :x.shape[1], :]

        # Apply bottleneck to each token embedding
        if self.bottleneck is not None:
            B, L, d = x.shape
            x_flat = x.view(-1, d)  # (B*L, d_model)
            x_bottleneck = self.bottleneck(x_flat)  # (B*L, bottleneck_dim)
            x = x_bottleneck.view(B, L, -1)  # (B, L, bottleneck_dim)

            if self.project_back is not None:
                x = self.project_back(x)  # (B, L, d_model)

        x = self.layer1(x)
        x = self.layer2(x)

        # Global pooling
        x = x.transpose(1, 2)  # (B, d_model, L)
        x = self.pool(x).squeeze(-1)  # (B, d_model)

        logits = self.classifier(x)  # (B, 2)
        return logits


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in tqdm(train_loader, desc="Training"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main():
    """Main Phase 4.2 experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Hyperparameters
    vocab_size = 10000
    seq_length = 100
    num_samples = 2000
    batch_size = 32
    num_epochs = 15
    learning_rate = 0.001
    d_model = 256
    n_heads = 4
    d_ff = 512
    bottleneck_dim = 128
    sig_g = torch.tensor([1.0, 1.0])  # Cl(2,0): 4D

    # Load synthetic SST-2-like data
    print("Loading SST-2-like sentiment data...")
    (X_train, y_train), (X_val, y_val) = load_sst2_simple(vocab_size, seq_length, num_samples)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Results dictionary
    all_results = {}

    # ========================
    # Baseline: Standard Transformer-2L
    # ========================
    print("\n" + "="*60)
    print("Baseline: Standard Transformer-2L")
    print("="*60)

    model_baseline = Transformer2L(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=learning_rate)

    baseline_train_losses = []
    baseline_train_accs = []
    baseline_val_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model_baseline, train_loader, criterion, optimizer_baseline, device)
        val_loss, val_acc = evaluate(model_baseline, val_loader, criterion, device)

        baseline_train_losses.append(train_loss)
        baseline_train_accs.append(train_acc)
        baseline_val_accs.append(val_acc)

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    all_results['baseline'] = {
        'final_val_accuracy': baseline_val_accs[-1],
        'train_accuracies': baseline_train_accs,
        'val_accuracies': baseline_val_accs,
        'train_losses': baseline_train_losses
    }

    # ========================
    # Clifford: Transformer-2L + P2.9 Bottleneck
    # ========================
    print("\n" + "="*60)
    print("Clifford: Transformer-2L + P2.9 Bottleneck")
    print("="*60)

    model_clifford = Transformer2LWithBottleneck(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        sig_g=sig_g,
        bottleneck_dim=bottleneck_dim
    ).to(device)

    optimizer_clifford = optim.Adam(model_clifford.parameters(), lr=learning_rate)

    clifford_train_losses = []
    clifford_train_accs = []
    clifford_val_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model_clifford, train_loader, criterion, optimizer_clifford, device)
        val_loss, val_acc = evaluate(model_clifford, val_loader, criterion, device)

        clifford_train_losses.append(train_loss)
        clifford_train_accs.append(train_acc)
        clifford_val_accs.append(val_acc)

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    all_results['clifford'] = {
        'final_val_accuracy': clifford_val_accs[-1],
        'train_accuracies': clifford_train_accs,
        'val_accuracies': clifford_val_accs,
        'train_losses': clifford_train_losses
    }

    # ========================
    # Summary and Comparison
    # ========================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    baseline_acc = all_results['baseline']['final_val_accuracy']
    clifford_acc = all_results['clifford']['final_val_accuracy']
    improvement = ((clifford_acc - baseline_acc) / baseline_acc) * 100

    print(f"\nBaseline (Standard Transformer-2L):")
    print(f"  Final Val Accuracy: {baseline_acc:.4f}")

    print(f"\nClifford (Transformer-2L + P2.9 Bottleneck):")
    print(f"  Final Val Accuracy: {clifford_acc:.4f}")

    print(f"\nImprovement:")
    print(f"  Absolute: {clifford_acc - baseline_acc:.4f}")
    print(f"  Relative: {improvement:.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/p4_2_transformer_sentiment_{timestamp}.json"

    import os
    os.makedirs("results", exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    results = main()
    print("\n✓ Phase 4.2 Complete: Language domain baseline established")
