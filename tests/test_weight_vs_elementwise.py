"""
Compare weight-based kernel call (which works) vs elementwise (which doesn't).
"""

import torch
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.cliffordkernels import get_3d_clifford_kernel

sig_g = torch.tensor([1.0, 1.0, 1.0])
sig = CliffordSignature(sig_g)
I = sig.n_blades  # 8

print("=== WEIGHT-BASED CASE (used in Linear layer style) ===")
# x: (B, Nin, I)
# y: (Nout, Nin, I) <- weights
B, Nin, Nout = 4, 3, 2

x_weight = torch.randn(B, Nin, I)
y_weight = torch.randn(Nout, Nin, I)

# w = y.permute(2, 0, 1) -> (I, Nout, Nin)
w_weight = y_weight.permute(2, 0, 1)
print(f"y shape (weights): {y_weight.shape}")
print(f"w shape (permuted): {w_weight.shape}")  # (I=8, Nout=2, Nin=3)

res = get_3d_clifford_kernel(w_weight, sig_g)
kernel_weight = res[1]
print(f"Kernel shape: {kernel_weight.shape}")
# For linear layer, this is used with F.linear(x_flat, kernel)
# So we'd flatten x to (B, Nin*I) and kernel should be (Nout*I, Nin*I) ??
print(f"  Expected for linear: ({Nout*I}, {Nin*I}) but got {kernel_weight.shape}")

print("\n=== ELEMENTWISE CASE (broken) ===")
# x: (B, N, I)
# y: (B, N, I)
B, N = 4, 3
BN = B * N

x_elem = torch.randn(B, N, I)
y_elem = torch.randn(B, N, I)

# Flatten to (BN, 1, I) then permute
x_flat = x_elem.reshape(BN, 1, I)
y_flat = y_elem.reshape(BN, 1, I)
w_elem = y_flat.permute(2, 0, 1)
print(f"y_flat shape: {y_flat.shape}")
print(f"w shape (permuted): {w_elem.shape}")  # (I=8, BN=12, 1)

res = get_3d_clifford_kernel(w_elem, sig_g)
kernel_elem = res[1]
print(f"Kernel shape: {kernel_elem.shape}")  # (96, 8) = (BN*I, I) = (12*8, 8)

print("\n=== THE PROBLEM ===")
print("The kernel function is designed for weight matrices of shape (Nout, Nin).")
print("It returns a kernel of shape (Nout*I, Nin*I) or similar.")
print()
print("In weight-based case: w is (I, Nout, Nin) - makes sense!")
print("  We're computing a 3D weight tensor")
print()
print("In elementwise case: w is (I, BN, 1) - WRONG!")
print("  We're treating BN batch elements as if they're Nout separate kernels!")
print("  And the single '1' dimension is treated as Nin, which is weird")
print()
print("The kernel output for elementwise [96, 8] = [BN*I, I] doesn't map cleanly")
print("to [BN, I, I] for batched matrix multiplication!")
