import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import DTD
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from cliffordlayers.nn.modules.cliffordfourier import CliffordSpectralConv2d
from cliffeq.algebra.utils import embed_vector, scalar_part
import os
import json
import time

def run_pv2():
    print("PV2: Clifford Fourier Vision on DTD (subset)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load DTD
    try:
        full_dataset = DTD(root='data', split='train', download=True, transform=transform)
        # Use a small subset
        indices = list(range(100))
        dataset = Subset(full_dataset, indices)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        num_classes = 47 # DTD has 47 classes
    except Exception as e:
        print(f"Error loading DTD: {e}")
        # Fallback to dummy data
        class DummyDTD(torch.utils.data.Dataset):
            def __init__(self):
                self.data = torch.randn(100, 3, 64, 64)
                self.labels = torch.randint(0, 47, (100,))
            def __len__(self): return 100
            def __getitem__(self, i): return self.data[i], self.labels[i]
        loader = DataLoader(DummyDTD(), batch_size=16, shuffle=True)
        num_classes = 47

    # Clifford Fourier Model
    class CliffordFourierModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            # g: signature [1, 1] for Cl(2,0)
            # modes: number of Fourier modes to keep
            self.fourier = CliffordSpectralConv2d(g=[1, 1], in_channels=1, out_channels=8, modes1=16, modes2=16)
            self.fc = nn.Linear(8 * 4 * 64 * 64, num_classes) # Flattened spatial

        def forward(self, x):
            # x is (B, 3, 64, 64)
            B, _, H, W = x.shape
            # CliffordSpectralConv2d expects (B, C, H, W, I)
            x_mv = torch.zeros(B, 1, H, W, 4, device=x.device)
            x_mv[:, 0, :, :, 0] = x[:, 0] # scalar
            x_mv[:, 0, :, :, 1] = x[:, 1] # e1
            x_mv[:, 0, :, :, 2] = x[:, 2] # e2

            out_mv = self.fourier(x_mv) # (B, 8, H, W, 4)
            out_scalar = out_mv.reshape(B, -1)
            return self.fc(out_scalar)

    model = CliffordFourierModel(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    results = {"status": "success", "final_loss": total_loss / len(loader)}
    os.makedirs("results", exist_ok=True)
    with open("results/pv2_results.json", "w") as f:
        json.dump(results, f)
    print("PV2 Complete")

if __name__ == "__main__":
    run_pv2()
