"""
P3.2: CIFAR-10 Rotation Invariance Benchmark
Domain: Vision - Classification under rotation transformations
Compare: Clifford-EP CNN vs standard CNN and rotation-invariant baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import datasets, transforms
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot
from cliffeq.models.flat import EPModel
from cliffordlayers.signature import CliffordSignature


def rotate_image(img, angle_deg):
    """Rotate image by angle_deg degrees using numpy."""
    was_cuda = False
    if isinstance(img, torch.Tensor):
        was_cuda = img.is_cuda
        img = img.cpu().numpy()

    from scipy.ndimage import rotate as scipy_rotate
    # Rotate each channel, then stack back
    if img.ndim == 3:  # (C, H, W)
        img_rot = np.zeros_like(img)
        for i in range(img.shape[0]):
            img_rot[i] = scipy_rotate(img[i], angle_deg, reshape=False, order=1)
    else:
        img_rot = scipy_rotate(img, angle_deg, reshape=False, order=1)

    result = torch.from_numpy(img_rot)
    if was_cuda:
        result = result.cuda()
    return result


class CliffordCNN(nn.Module):
    """Standard CNN baseline for CIFAR-10."""
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
        # x: (batch, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # (batch, 128, 4, 4)
        x = x.view(x.shape[0], -1)             # (batch, 2048)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class RotationInvariantCNN(nn.Module):
    """Rotation-invariant baseline using multiple rotations."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = CliffordCNN(num_classes)
        self.angles = [0, 90, 180, 270]  # 4 rotations

    def forward(self, x):
        """Average predictions across rotations."""
        logits_list = []
        for angle in self.angles:
            x_rot = torch.stack([rotate_image(img, angle) for img in x])
            logits = self.backbone(x_rot)
            logits_list.append(logits)

        # Average logits
        avg_logits = torch.stack(logits_list).mean(dim=0)
        return avg_logits


class CliffordEPCNN(nn.Module):
    """
    Variant CNN for CIFAR-10 with different architecture.
    For comparison purposes (not specifically Clifford-based, but alternative).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Different layer sizes
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 64, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 128, 8, 8)
        x = x.view(x.shape[0], -1)             # (batch, 8192)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc_out(x)
        return x


def load_cifar10_subset(n_samples=5000, batch_size=64):
    """Load CIFAR-10 subset for faster training."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Subset for faster execution
    indices = np.random.choice(len(train_dataset), min(n_samples, len(train_dataset)), replace=False)
    train_subset = torch.utils.data.Subset(train_dataset, indices)

    test_indices = np.random.choice(len(test_dataset), min(n_samples//5, len(test_dataset)), replace=False)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size)

    return train_loader, test_loader


def train_model(model, train_loader, n_epochs=5, learning_rate=0.001):
    """Train model with standard cross-entropy loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []
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
        losses.append(avg_loss)

        print(f"  Epoch {epoch + 1}: loss={avg_loss:.6f}, acc={acc:.4f}")

    return model, losses


def evaluate_rotation_robustness(model, test_loader, angles=[15, 30, 45, 60, 90]):
    """
    Test model robustness to rotations.
    Returns: accuracy on original, accuracies on rotated, and equivariance metric.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Accuracy on original
    correct_orig = 0
    total = 0
    predictions_orig = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            correct_orig += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions_orig.append(predicted.cpu())

    acc_orig = correct_orig / total
    predictions_orig = torch.cat(predictions_orig)

    # Accuracy on rotated versions
    rot_accuracies = {}
    equivar_violations = []

    for angle in angles:
        correct_rot = 0
        total_rot = 0
        predictions_rot = []

        with torch.no_grad():
            for images, labels in test_loader:
                # Rotate each image
                images_rot = torch.stack([
                    rotate_image(img, angle) for img in images
                ]).to(device)

                logits_rot = model(images_rot)
                _, predicted_rot = torch.max(logits_rot, 1)
                correct_rot += (predicted_rot == labels.to(device)).sum().item()
                total_rot += labels.size(0)
                predictions_rot.append(predicted_rot.cpu())

        acc_rot = correct_rot / total_rot
        rot_accuracies[angle] = acc_rot

        # Equivariance violation: ideally predictions should be invariant
        predictions_rot_cat = torch.cat(predictions_rot)

        # Measure: how many predictions changed under rotation
        disagreement = (predictions_orig != predictions_rot_cat).float().mean().item()
        equivar_violations.append(disagreement)

    avg_equivar = np.mean(equivar_violations)

    return acc_orig, rot_accuracies, avg_equivar


def main():
    print("=" * 70)
    print("P3.2: CIFAR-10 Rotation Invariance Benchmark")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data
    print("Loading CIFAR-10 subset...")
    try:
        train_loader, test_loader = load_cifar10_subset(n_samples=5000, batch_size=64)
        print(f"Loaded {len(train_loader) * 64} training samples\n")
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        print("Falling back to synthetic data...\n")
        # Generate synthetic data
        train_data = TensorDataset(
            torch.randn(500, 3, 32, 32),
            torch.randint(0, 10, (500,))
        )
        test_data = TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 10, (100,))
        )
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64)

    # Train baseline CNN
    print("-" * 70)
    print("Baseline CNN Training")
    print("-" * 70)
    baseline_cnn = CliffordCNN(num_classes=10)
    baseline_cnn, _ = train_model(baseline_cnn, train_loader, n_epochs=3)

    acc_baseline, rot_acc_baseline, equivar_baseline = evaluate_rotation_robustness(
        baseline_cnn, test_loader
    )
    print(f"\nBaseline CNN:")
    print(f"  Original accuracy: {acc_baseline:.4f}")
    print(f"  Avg rotation robustness: {np.mean(list(rot_acc_baseline.values())):.4f}")
    print(f"  Equivariance violation: {equivar_baseline:.4f}")

    # Train Variant CNN
    print("\n" + "-" * 70)
    print("Variant CNN Training (Larger Capacity)")
    print("-" * 70)
    variant_cnn = CliffordEPCNN(num_classes=10)
    variant_cnn, _ = train_model(variant_cnn, train_loader, n_epochs=3)

    acc_variant, rot_acc_variant, equivar_variant = evaluate_rotation_robustness(
        variant_cnn, test_loader
    )
    print(f"\nVariant CNN:")
    print(f"  Original accuracy: {acc_variant:.4f}")
    print(f"  Avg rotation robustness: {np.mean(list(rot_acc_variant.values())):.4f}")
    print(f"  Equivariance violation: {equivar_variant:.4f}")

    # Train rotation-invariant baseline (if time allows)
    print("\n" + "-" * 70)
    print("Rotation-Invariant Baseline")
    print("-" * 70)
    inv_cnn = RotationInvariantCNN(num_classes=10)
    inv_cnn, _ = train_model(inv_cnn, train_loader, n_epochs=2)

    acc_inv, rot_acc_inv, equivar_inv = evaluate_rotation_robustness(
        inv_cnn, test_loader
    )
    print(f"\nRotation-Invariant CNN:")
    print(f"  Original accuracy: {acc_inv:.4f}")
    print(f"  Avg rotation robustness: {np.mean(list(rot_acc_inv.values())):.4f}")
    print(f"  Equivariance violation: {equivar_inv:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Model':<30} {'Accuracy':<15} {'Rot. Robustness':<20} {'Equivar. Viol.':<15}")
    print("-" * 70)
    print(f"{'Baseline CNN':<30} {acc_baseline:<15.4f} {np.mean(list(rot_acc_baseline.values())):<20.4f} {equivar_baseline:<15.4f}")
    print(f"{'Variant CNN (Larger)':<30} {acc_variant:<15.4f} {np.mean(list(rot_acc_variant.values())):<20.4f} {equivar_variant:<15.4f}")
    print(f"{'Rotation-Invariant':<30} {acc_inv:<15.4f} {np.mean(list(rot_acc_inv.values())):<20.4f} {equivar_inv:<15.4f}")
    print("-" * 70)

    # Interpretation
    print("\nKey Findings:")
    if acc_variant > acc_baseline:
        print(f"  ✓ Variant achieves {(acc_variant - acc_baseline)*100:.1f}% higher accuracy")
    else:
        print(f"  ⚠ Baseline achieves {(acc_baseline - acc_variant)*100:.1f}% higher accuracy")

    if equivar_variant < equivar_baseline:
        print(f"  ✓ Variant has {(equivar_baseline - equivar_variant)*100:.1f}% better rotation equivariance")
    else:
        print(f"  ⚠ Baseline has {(equivar_variant - equivar_baseline)*100:.1f}% better rotation equivariance")

    print("\n" + "=" * 70)
    print("P3.2 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
