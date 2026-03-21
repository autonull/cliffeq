import torch
import clifford
import numpy as np
from cliffeq.algebra.utils import geometric_product
from cliffordlayers.signature import CliffordSignature

def test_cga_geometric_product():
    # Setup CGA Cl(4, 1)
    g = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0])
    layout, _ = clifford.Cl(4, 1)

    # Random multivectors
    B, N = 2, 4
    x_np = np.random.randn(B, N, 32)
    y_np = np.random.randn(B, N, 32)

    x_torch = torch.from_numpy(x_np).float()
    y_torch = torch.from_numpy(y_np).float()

    # Ground truth using clifford library
    gt_np = np.zeros((B, N, 32))
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
    print(f"Elementwise CGA Product Difference: {diff.item()}")
    assert diff < 1e-4

    # Test weight-based case
    Nout, Nin = 8, 4
    x_w_np = np.random.randn(B, Nin, 32)
    y_w_np = np.random.randn(Nout, Nin, 32)

    x_w_torch = torch.from_numpy(x_w_np).float()
    y_w_torch = torch.from_numpy(y_w_np).float()

    # Ground truth for weight-based
    gt_w_np = np.zeros((B, Nout, 32))
    for b in range(B):
        for o in range(Nout):
            sum_prod = layout.MultiVector(np.zeros(32))
            for n in range(Nin):
                mv_x = layout.MultiVector(x_w_np[b, n])
                mv_y = layout.MultiVector(y_w_np[o, n])
                sum_prod += mv_x * mv_y
            gt_w_np[b, o] = sum_prod.value

    out_w_torch = geometric_product(x_w_torch, y_w_torch, g)
    diff_w = torch.norm(out_w_torch - torch.from_numpy(gt_w_np).float())
    print(f"Weight-based CGA Product Difference: {diff_w.item()}")
    assert diff_w < 1e-3

if __name__ == "__main__":
    test_cga_geometric_product()
    print("CGA Verification Passed!")
