import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import json
from cliffeq.models.hybrid import CliffordEPBottleneck, CliffordBPBottleneck
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot

def create_synthetic_data(num=100):
    dataset = []
    for _ in range(num):
        G = nx.erdos_renyi_graph(20, 0.2)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        if edge_index.numel() == 0: edge_index = torch.zeros((2, 0), dtype=torch.long)
        x = torch.randn(20, 8); y = torch.tensor([1 if nx.density(G) > 0.2 else 0])
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset

class GCNBottleneck(nn.Module):
    def __init__(self, variant="baseline", in_c=8, hidden_c=32):
        super().__init__()
        self.variant = variant
        self.conv1 = GCNConv(in_c, hidden_c)
        if variant == "clifford-ep":
            self.bottleneck = CliffordEPBottleneck(BilinearEnergy(8, 8, torch.tensor([1., 1.])), LinearDot(), comp=4)
            self.proj_back = nn.Linear(8, hidden_c)
        elif variant == "clifford-bp":
            self.bottleneck = CliffordBPBottleneck(hidden_c, 8, torch.tensor([1., 1.]))
        self.conv2 = GCNConv(hidden_c, hidden_c)
        self.classifier = nn.Linear(hidden_c, 2)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        if self.variant == "clifford-ep":
            x = F.relu(self.proj_back(self.bottleneck(x)))
        elif self.variant == "clifford-bp":
            x = F.relu(self.bottleneck(x))
        return self.classifier(global_mean_pool(F.relu(self.conv2(x, edge_index)), batch))

def train_eval(variant):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(create_synthetic_data(100), batch_size=32, shuffle=True)
    model = GCNBottleneck(variant).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(data.x, data.edge_index, data.batch), data.y).backward()
            optimizer.step()
    test_data = create_synthetic_data(50)
    correct = 0
    model.eval()
    with torch.no_grad():
        for data in test_data:
            data = data.to(device)
            if model(data.x, data.edge_index, data.batch).argmax() == data.y: correct += 1
    return correct / 50

def main():
    print("Graph Domain Ablation")
    results = {}
    for var in ["baseline", "clifford-ep", "clifford-bp"]:
        print(f"  Running {var}...")
        results[var] = train_eval(var)
        print(f"    Acc: {results[var]:.4f}")
    with open("results/p4_4_gnn_graph.json", "w") as f: json.dump(results, f)

if __name__ == "__main__": main()
