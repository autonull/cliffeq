import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from cliffeq.models.bottleneck_v4_adaptive import CliffordAdaptiveBottleneck
import os
import json
import time

def load_cifar10_subset(n_samples=5000, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_indices = torch.randperm(len(train_dataset))[:n_samples]
    test_indices = torch.randperm(len(test_dataset))[:1000]

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class ResNetWithBottleneck(nn.Module):
    def __init__(self, use_bottleneck=False):
        super().__init__()
        from torchvision.models import resnet18
        self.resnet = resnet18(num_classes=10)
        self.use_bottleneck = use_bottleneck

        if use_bottleneck:
            # Insert after layer2
            # layer2 output is (B, 128, 8, 8)
            sig_g = torch.tensor([1.0, 1.0]) # Cl(2,0)
            self.bottleneck = CliffordAdaptiveBottleneck(128, 16, sig_g)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)

        if self.use_bottleneck:
            # (B, 128, 8, 8) -> permute -> (B, 8, 8, 128)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
            x = self.bottleneck(x)
            x = x.permute(0, 3, 1, 2)

        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def run_experiment():
    print("="*80)
    print("P4: ResNet-18 with Adaptive Clifford Bottleneck on CIFAR-10")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cifar10_subset(n_samples=2000)

    results = {}

    for name, use_bottleneck in [("Baseline", False), ("Adaptive Clifford", True)]:
        print(f"\nTraining {name}...")
        model = ResNetWithBottleneck(use_bottleneck=use_bottleneck).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(5):
            loss, acc = train(model, train_loader, optimizer, device)
            print(f"  Epoch {epoch+1}, Loss: {loss:.4f}, Train Acc: {acc:.4f}")

        test_acc = evaluate(model, test_loader, device)
        print(f"  Test Accuracy: {test_acc:.4f}")
        results[name] = test_acc

    os.makedirs("results", exist_ok=True)
    with open("results/p4_adaptive_bottleneck_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_experiment()
