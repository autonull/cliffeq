import torch
import clifford
import numpy as np
from cliffeq.algebra.utils import geometric_product

def test_cl13_geometric_product():
    # Setup Cl(1, 3)
    g = torch.tensor([1.0, -1.0, -1.0, -1.0])
    layout, _ = clifford.Cl(1, 3)

    # Random multivectors
    B, N = 2, 4
    x_np = np.random.randn(B, N, 16)
    y_np = np.random.randn(B, N, 16)

    x_torch = torch.from_numpy(x_np).float()
    y_torch = torch.from_numpy(y_np).float()

    # Ground truth using clifford library
    gt_np = np.zeros((B, N, 16))
    for b in range(B):
        for n in range(N):
            mv_x = layout.MultiVector(x_np[b, n])
            mv_y = layout.MultiVector(y_np[b, n])
            prod = mv_x * mv_y
            gt_np[b, n] = prod.value

    # Implementation under test
    out_torch = geometric_product(x_torch, y_torch, g)

    # Compare
    diff = torch.norm(out_torch - torch.from_numpy(gt_np).float())
    print(f"Elementwise Cl(1,3) Product Difference: {diff.item()}")
    assert diff < 1e-4

if __name__ == "__main__":
    test_cl13_geometric_product()
    print("Cl(1,3) Verification Passed!")
