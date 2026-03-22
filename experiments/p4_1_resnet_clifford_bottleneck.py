import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import json
import os
import numpy as np
from cliffeq.models.hybrid import CliffordEPBottleneck, CliffordBPBottleneck
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot

def load_data(n_samples=500):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_idx = torch.randperm(len(train_set))[:n_samples]
    test_idx = torch.randperm(len(test_set))[:100]
    return DataLoader(Subset(train_set, train_idx), batch_size=32, shuffle=True), DataLoader(Subset(test_set, test_idx), batch_size=32)

class ResNetBottleneck(nn.Module):
    def __init__(self, variant="baseline"):
        super().__init__()
        from torchvision.models import resnet18
        self.resnet = resnet18(num_classes=10)
        self.variant = variant
        if variant == "clifford-ep":
            g = torch.tensor([1.0, 1.0])
            energy = BilinearEnergy(16, 16, g)
            self.bottleneck = CliffordEPBottleneck(energy, LinearDot(), n_free=5, comp=4)
            self.proj_back = nn.Linear(16, 64)
        elif variant == "clifford-bp":
            self.bottleneck = CliffordBPBottleneck(64, 16, torch.tensor([1.0, 1.0]))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        if self.variant == "clifford-ep":
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, C)
            x = self.bottleneck(x)
            x = self.proj_back(x).view(B, H, W, C).permute(0, 3, 1, 2)
        elif self.variant == "clifford-bp":
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, C)
            x = self.bottleneck(x).view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return self.resnet.fc(x)

def train_eval(variant, train_loader, test_loader, device):
    model = ResNetBottleneck(variant).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(images), labels).backward()
            optimizer.step()

    correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(test_loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()
    results = {}
    for var in ["baseline", "clifford-ep", "clifford-bp"]:
        print(f"Running {var}...")
        results[var] = train_eval(var, train_loader, test_loader, device)
        print(f"  Accuracy: {results[var]:.4f}")
    os.makedirs("results", exist_ok=True)
    with open("results/p4_1_resnet_cifar10.json", "w") as f: json.dump(results, f)

if __name__ == "__main__": main()
