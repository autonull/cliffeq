"""
Inspect the individual kernel matrices.
"""

import torch
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.cliffordkernels import get_3d_clifford_kernel

# Setup
sig_g = torch.tensor([1.0, 1.0, 1.0])
sig = CliffordSignature(sig_g)
I = sig.n_blades  # 8

B, N = 4, 3
BN = B * N

# Create test inputs: scalars 2.0 * 3.0 = 6.0
x = torch.zeros(B, N, I)
y = torch.zeros(B, N, I)
x[..., 0] = 2.0  
y[..., 0] = 3.0  

x_flat = x.reshape(BN, 1, I)
y_flat = y.reshape(BN, 1, I)

# Permute y as weights
w = y_flat.permute(2, 0, 1)  # (I, BN, 1)

# Get kernel
res = get_3d_clifford_kernel(w, sig_g)
kernel = res[1]

# Reshape to [12, 8, 8]
kernel_reshaped = kernel.view(BN, I, I)

print("Kernel matrices for each batch element:")
for i in range(BN):
    K = kernel_reshaped[i]
    print(f"\nBatch {i} (kernel[{i}]):")
    print(f"  Shape: {K.shape}")
    print(f"  First row (for y=3, x=2 -> 6): {K[0]}")
    print(f"  Sum of first col: {K[:, 0].sum()}")
    print(f"  Nonzero rows: {(K.sum(dim=1) != 0).sum().item()}")
    if i < 3:
        print(f"  Full matrix:\n{K}")
