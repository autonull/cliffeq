"""
Inspect cliffordlayers kernel output to understand its format.
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
x[..., 0] = 2.0  # All scalars are 2.0
y[..., 0] = 3.0  # All scalars are 3.0

x_flat = x.reshape(BN, 1, I)
y_flat = y.reshape(BN, 1, I)

print(f"x_flat shape: {x_flat.shape}")
print(f"y_flat shape: {y_flat.shape}")
print(f"x_flat (all zeros except [i, 0, 0] = 2.0):")
print(f"  x_flat[0]: {x_flat[0, :, :]}")
print(f"  x_flat[1]: {x_flat[1, :, :]}")

# Permute y as weights
w = y_flat.permute(2, 0, 1)  # (I, BN, 1)
print(f"\nw (weights) shape: {w.shape}")
print(f"w[:, 0, 0] (blade 0 for all batch elements): {w[:, 0, 0]}")
print(f"w[0, :, 0] (all batch elements for blade 0): {w[0, :, 0]}")

# Get kernel
res = get_3d_clifford_kernel(w, sig_g)
kernel = res[1] if isinstance(res, tuple) else res
print(f"\nKernel is tuple: {isinstance(res, tuple)}")
if isinstance(res, tuple):
    print(f"  Tuple length: {len(res)}")
    print(f"  res[0] shape: {res[0].shape if hasattr(res[0], 'shape') else type(res[0])}")
    print(f"  res[1] shape: {res[1].shape}")
print(f"Kernel shape: {kernel.shape}")
print(f"Kernel (first 20 rows, all 8 cols):\n{kernel[:20, :]}")

# Now reshape and see what happens
kernel_reshaped = kernel.view(BN, I, I)
print(f"\nKernel reshaped: {kernel_reshaped.shape}")

# Apply kernel to x
x_reshaped = x_flat.view(BN, I)
print(f"\nx_reshaped shape: {x_reshaped.shape}")
print(f"x_reshaped[0]: {x_reshaped[0]}")
print(f"x_reshaped[1]: {x_reshaped[1]}")

# Apply kernel for first few batch elements
for i in range(min(3, BN)):
    result_i = torch.matmul(kernel_reshaped[i], x_reshaped[i])
    print(f"\nBatch {i}:")
    print(f"  kernel[{i}] @ x[{i}] = {result_i}")
    print(f"  Expected: [6, 6, 6, 6, 6, 6, 6, 6]")

# Check if kernel structure is correct
print("\n\n=== Analyzing Kernel Structure ===")
print("If kernel is [96, 8] = [12*8, 8], reshaped to [12, 8, 8]:")
print("  Each of 12 batch elements gets 8x8 matrix")
print("  kernel[0] should multiply input 0 to produce output 0")
print("  kernel[1] should multiply input 1 to produce output 1")

# Test full bmm
out_full = torch.bmm(kernel_reshaped, x_reshaped.unsqueeze(-1)).squeeze(-1)
print(f"\nFull bmm output shape: {out_full.shape}")
print(f"Output reshaped to [4, 3, 8]:")
out = out_full.view(B, N, I)
print(f"out[0, :, 0] (should all be 6.0): {out[0, :, 0]}")
print(f"out[1, :, 0] (should all be 6.0): {out[1, :, 0]}")
