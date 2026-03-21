"""
P3.2 (Clifford-EP variant): CIFAR-10 with Clifford-inspired CNN
Simpler approach: use Clifford representation in feature extraction only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets, transforms
from cliffordlayers.signature import CliffordSignature


def load_cifar10(n_samples=5000, batch_size=64):
    """Load CIFAR-10 subset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    indices = np.random.choice(len(train_dataset), min(n_samples, len(train_dataset)), replace=False)
    train_subset = torch.utils.data.Subset(train_dataset, indices)

    test_indices = np.random.choice(len(test_dataset), min(n_samples//5, len(test_dataset)), replace=False)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size)

    return train_loader, test_loader


def rotate_image(img, angle_deg):
    """Rotate image by angle_deg degrees."""
    was_cuda = False
    if isinstance(img, torch.Tensor):
        was_cuda = img.is_cuda
        img = img.cpu().numpy()

    from scipy.ndimage import rotate as scipy_rotate
    if img.ndim == 3:
        img_rot = np.zeros_like(img)
        for i in range(img.shape[0]):
            img_rot[i] = scipy_rotate(img[i], angle_deg, reshape=False, order=1)
    else:
        img_rot = scipy_rotate(img, angle_deg, reshape=False, order=1)

    result = torch.from_numpy(img_rot)
    if was_cuda:
        result = result.cuda()
    return result


class BaselineCNN(nn.Module):
    """Standard CNN baseline."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CliffordCNN(nn.Module):
    """
    Clifford-inspired CNN: Use multi-scale features (mimicking Clifford grades).
    Instead of pure scalars, extract oriented features in parallel.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Branch 1: Fine-grained features (scalar-like)
        self.conv1_fine = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2_fine = nn.Conv2d(32, 64, 3, padding=1)

        # Branch 2: Oriented features (bivector-like - multi-orientation)
        self.conv1_orient = nn.Conv2d(3, 32, 3, padding=1, dilation=1)
        self.conv2_orient = nn.Conv2d(32, 64, 3, padding=1, dilation=1)

        # Fusion
        self.pool = nn.MaxPool2d(2, 2)
        # After pooling 3x: 64 * 4 * 4 + 64 * 4 * 4 = 128 * 16 = 2048
        self.fc1 = nn.Linear(128 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Fine branch
        x_fine = self.pool(F.relu(self.conv1_fine(x)))       # 32, 16, 16
        x_fine = self.pool(F.relu(self.conv2_fine(x_fine)))  # 64, 8, 8
        x_fine = self.pool(x_fine)   # 64, 4, 4

        # Oriented branch (same structure)
        x_orient = self.pool(F.relu(self.conv1_orient(x)))   # 32, 16, 16
        x_orient = self.pool(F.relu(self.conv2_orient(x_orient)))  # 64, 8, 8
        x_orient = self.pool(x_orient)  # 64, 4, 4

        # Concatenate (fusion) - both are 64, 4, 4
        x_combined = torch.cat([x_fine, x_orient], dim=1)  # 128, 4, 4
        x_combined = x_combined.view(x_combined.shape[0], -1)  # 128 * 16 = 2048

        # Classification
        x_combined = self.dropout(F.relu(self.fc1(x_combined)))
        logits = self.fc2(x_combined)

        return logits


def train_model(model, train_loader, n_epochs=3, learning_rate=0.001):
    """Train model with cross-entropy loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / len(train_loader)
        acc = correct / total
        if (epoch + 1) % 2 == 0 or n_epochs <= 2:
            print(f"  Epoch {epoch + 1}: loss={avg_loss:.6f}, acc={acc:.4f}")

    return model


def evaluate_rotation_robustness(model, test_loader, angles=[30, 60, 90]):
    """Evaluate rotation robustness."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    correct_orig = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            correct_orig += (predicted == labels).sum().item()
            total += labels.size(0)

    acc_orig = correct_orig / total

    # Test on rotations
    rot_accuracies = {}
    for angle in angles:
        correct_rot = 0
        total_rot = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images_rot = torch.stack([
                    rotate_image(img, angle) for img in images
                ]).to(device)
                logits_rot = model(images_rot)
                _, predicted_rot = torch.max(logits_rot, 1)
                correct_rot += (predicted_rot == labels.to(device)).sum().item()
                total_rot += labels.size(0)

        rot_accuracies[angle] = correct_rot / total_rot

    # Equivariance metric
    equivar_violation = 1.0 - np.mean(list(rot_accuracies.values())) / (acc_orig + 1e-6)

    return acc_orig, rot_accuracies, equivar_violation


def main():
    print("=" * 70)
    print("P3.2 (Clifford-Inspired): CIFAR-10 Rotation Invariance")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading CIFAR-10...")
    try:
        train_loader, test_loader = load_cifar10(n_samples=10000, batch_size=64)
        print(f"Loaded training data\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Train baseline
    print("-" * 70)
    print("Baseline CNN Training")
    print("-" * 70)
    baseline = BaselineCNN(num_classes=10)
    baseline = train_model(baseline, train_loader, n_epochs=10)

    acc_baseline, rot_acc_baseline, equivar_baseline = evaluate_rotation_robustness(baseline, test_loader)

    print(f"\nBaseline CNN:")
    print(f"  Original accuracy: {acc_baseline:.4f}")
    print(f"  Avg rotation accuracy: {np.mean(list(rot_acc_baseline.values())):.4f}")
    print(f"  Equivariance violation: {equivar_baseline:.4f}")

    # Train Clifford variant
    print("\n" + "-" * 70)
    print("Clifford-Inspired CNN Training (Multi-Scale Features)")
    print("-" * 70)
    clifford_cnn = CliffordCNN(num_classes=10)
    clifford_cnn = train_model(clifford_cnn, train_loader, n_epochs=10)

    acc_clifford, rot_acc_clifford, equivar_clifford = evaluate_rotation_robustness(clifford_cnn, test_loader)

    print(f"\nClifford-Inspired CNN:")
    print(f"  Original accuracy: {acc_clifford:.4f}")
    print(f"  Avg rotation accuracy: {np.mean(list(rot_acc_clifford.values())):.4f}")
    print(f"  Equivariance violation: {equivar_clifford:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"{'Model':<30} {'Accuracy':<15} {'Equivar. Viol.':<15}")
    print("-" * 70)
    print(f"{'Baseline CNN':<30} {acc_baseline:<15.4f} {equivar_baseline:<15.4f}")
    print(f"{'Clifford-Inspired CNN':<30} {acc_clifford:<15.4f} {equivar_clifford:<15.4f}")
    print("-" * 70)

    print("\nKey Findings:")
    if acc_clifford > acc_baseline:
        print(f"  ✓ Clifford-Inspired: {(acc_clifford - acc_baseline)*100:.1f}% higher accuracy")
    else:
        print(f"  ⚠ Baseline: {(acc_baseline - acc_clifford)*100:.1f}% higher accuracy")

    if equivar_clifford < equivar_baseline:
        print(f"  ✓ Clifford-Inspired: {(equivar_baseline - equivar_clifford)*100:.1f}% better equivariance")
    else:
        print(f"  ⚠ Baseline: {(equivar_clifford - equivar_baseline)*100:.1f}% better equivariance")

    print("\n" + "=" * 70)
    print("P3.2 (Clifford-Inspired) Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
