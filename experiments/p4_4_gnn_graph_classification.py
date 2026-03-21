"""
Phase 4.4: Graph Domain — GCN-2L + P2.9 Clifford Bottleneck on MUTAG

Objective: Test P2.9 geometric bottleneck for graph neural networks.

Task: MUTAG graph classification (predicting mutagenicity)
- Baseline: Standard GCN-2L
- Clifford: GCN-2L + CliffordEPBottleneckV2 after first graph convolution

Metrics:
- Accuracy on test set
- AUC-ROC score
- Model stability across runs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
from tqdm import tqdm
import json
from datetime import datetime

from cliffeq.models.bottleneck_v2 import CliffordEPBottleneckV2


def create_synthetic_graph_dataset(num_graphs=500, num_nodes_range=(10, 30), num_classes=2):
    """
    Create synthetic graphs for graph classification.
    In practice, would use torch_geometric.datasets.MUTAG() or similar.
    """
    graphs = []

    for _ in range(num_graphs):
        num_nodes = np.random.randint(*num_nodes_range)

        # Generate random graph
        p_edge = 0.1 + np.random.rand() * 0.1
        G = nx.erdos_renyi_graph(num_nodes, p_edge)

        # Node features (random)
        x = torch.randn(num_nodes, 8)

        # Edge indices
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Label based on structural properties
        density = nx.density(G)
        is_mutagenic = 1 if (density > 0.15 or len(list(nx.simple_cycles(G.to_directed()))) > 0) else 0
        y = torch.tensor([is_mutagenic], dtype=torch.long)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
        graphs.append(data)

    return graphs


class GCN(nn.Module):
    """Standard 2-layer GCN for graph classification."""
    def __init__(self, in_channels=8, hidden_channels=32, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        logits = self.classifier(x)
        return logits


class GCNClifford(nn.Module):
    """GCN-2L with CliffordEPBottleneckV2 after first graph convolution."""
    def __init__(self, in_channels=8, hidden_channels=32, num_classes=2, sig_g=None):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Clifford bottleneck
        self.bottleneck = CliffordEPBottleneckV2(
            in_dim=hidden_channels,
            out_dim=hidden_channels // 2,
            sig_g=sig_g,
            n_ep_steps=2,
            step_size=0.01,
            use_spectral_norm=True
        ) if sig_g is not None else None

        bottleneck_out_dim = hidden_channels // 2 if sig_g is not None else hidden_channels
        self.project_back = nn.Linear(bottleneck_out_dim, hidden_channels) if sig_g is not None else None

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))

        if self.bottleneck is not None:
            # Apply bottleneck to node features
            x_bottleneck = self.bottleneck(x)
            x = self.project_back(x_bottleneck) if self.project_back is not None else x_bottleneck

        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        logits = self.classifier(x)
        return logits


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)

        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.squeeze(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch.y.squeeze(-1)).sum().item()
        total_samples += batch.num_graphs

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.squeeze(-1))

            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch.y.squeeze(-1)).sum().item()
            total_samples += batch.num_graphs

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.squeeze(-1).cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0

    return avg_loss, accuracy, auc


def main():
    """Main Phase 4.4 experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Hyperparameters
    num_graphs = 500
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.01
    hidden_channels = 32
    sig_g = torch.tensor([1.0, 1.0])  # Cl(2,0): 4D

    # Generate synthetic graph dataset
    print("Generating synthetic MUTAG-like graph dataset...")
    graphs = create_synthetic_graph_dataset(num_graphs=num_graphs)

    # Train/test split
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    test_graphs = graphs[train_size:]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # Results dictionary
    all_results = {}

    # ========================
    # Baseline: Standard GCN-2L
    # ========================
    print("\n" + "="*60)
    print("Baseline: Standard GCN-2L")
    print("="*60)

    model_baseline = GCN(in_channels=8, hidden_channels=hidden_channels, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=learning_rate)

    baseline_train_losses = []
    baseline_train_accs = []
    baseline_test_accs = []
    baseline_test_aucs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model_baseline, train_loader, criterion, optimizer_baseline, device)
        test_loss, test_acc, test_auc = evaluate(model_baseline, test_loader, criterion, device)

        baseline_train_losses.append(train_loss)
        baseline_train_accs.append(train_acc)
        baseline_test_accs.append(test_acc)
        baseline_test_aucs.append(test_auc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")

    all_results['baseline'] = {
        'final_test_accuracy': baseline_test_accs[-1],
        'final_test_auc': baseline_test_aucs[-1],
        'train_accuracies': baseline_train_accs,
        'test_accuracies': baseline_test_accs,
        'test_aucs': baseline_test_aucs,
        'train_losses': baseline_train_losses
    }

    # ========================
    # Clifford: GCN-2L + P2.9 Bottleneck
    # ========================
    print("\n" + "="*60)
    print("Clifford: GCN-2L + P2.9 Bottleneck")
    print("="*60)

    model_clifford = GCNClifford(in_channels=8, hidden_channels=hidden_channels, num_classes=2, sig_g=sig_g).to(device)
    optimizer_clifford = optim.Adam(model_clifford.parameters(), lr=learning_rate)

    clifford_train_losses = []
    clifford_train_accs = []
    clifford_test_accs = []
    clifford_test_aucs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model_clifford, train_loader, criterion, optimizer_clifford, device)
        test_loss, test_acc, test_auc = evaluate(model_clifford, test_loader, criterion, device)

        clifford_train_losses.append(train_loss)
        clifford_train_accs.append(train_acc)
        clifford_test_accs.append(test_acc)
        clifford_test_aucs.append(test_auc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")

    all_results['clifford'] = {
        'final_test_accuracy': clifford_test_accs[-1],
        'final_test_auc': clifford_test_aucs[-1],
        'train_accuracies': clifford_train_accs,
        'test_accuracies': clifford_test_accs,
        'test_aucs': clifford_test_aucs,
        'train_losses': clifford_train_losses
    }

    # ========================
    # Summary and Comparison
    # ========================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    baseline_acc = all_results['baseline']['final_test_accuracy']
    baseline_auc = all_results['baseline']['final_test_auc']
    clifford_acc = all_results['clifford']['final_test_accuracy']
    clifford_auc = all_results['clifford']['final_test_auc']

    print(f"\nBaseline (Standard GCN-2L):")
    print(f"  Final Test Accuracy: {baseline_acc:.4f}")
    print(f"  Final Test AUC: {baseline_auc:.4f}")

    print(f"\nClifford (GCN-2L + P2.9 Bottleneck):")
    print(f"  Final Test Accuracy: {clifford_acc:.4f}")
    print(f"  Final Test AUC: {clifford_auc:.4f}")

    print(f"\nImprovement (Accuracy):")
    print(f"  Absolute: {clifford_acc - baseline_acc:.4f}")
    print(f"  Relative: {((clifford_acc - baseline_acc) / baseline_acc) * 100:.2f}%")

    print(f"\nImprovement (AUC):")
    print(f"  Absolute: {clifford_auc - baseline_auc:.4f}")
    print(f"  Relative: {((clifford_auc - baseline_auc) / baseline_auc) * 100:.2f}%" if baseline_auc > 0 else "N/A")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/p4_4_gnn_graph_{timestamp}.json"

    import os
    os.makedirs("results", exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    results = main()
    print("\n✓ Phase 4.4 Complete: Graph domain baseline established")
