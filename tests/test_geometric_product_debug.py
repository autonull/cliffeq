"""
Debug geometric_product to understand the bug.
"""

import torch
from cliffeq.algebra.utils import geometric_product
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.cliffordkernels import get_3d_clifford_kernel


def debug_geometric_product():
    """Trace through geometric_product to find the bug."""
    sig_g = torch.tensor([1.0, 1.0, 1.0])
    sig = CliffordSignature(sig_g)
    I = sig.n_blades  # Should be 8

    B, N = 4, 3

    # Two scalars: 2 * 3 = 6
    x = torch.zeros(B, N, I)
    y = torch.zeros(B, N, I)
    x[..., 0] = 2.0
    y[..., 0] = 3.0

    print(f"Input shapes: x={x.shape}, y={y.shape}")
    print(f"Clifford dimension: {sig.dim}, n_blades: {sig.n_blades}")

    # Manually trace through the function
    # Elementwise case: x.shape == y.shape
    B_manual, N_manual, I_x = x.shape
    print(f"\nManual trace:")
    print(f"  B={B_manual}, N={N_manual}, I={I_x}")

    # Flatten
    x_flat = x.reshape(B_manual * N_manual, 1, I_x)
    y_flat = y.reshape(B_manual * N_manual, 1, I_x)
    print(f"  After reshape: x_flat={x_flat.shape}, y_flat={y_flat.shape}")

    # Permute y to weights
    w = y_flat.permute(2, 0, 1)
    print(f"  After permute: w={w.shape}")  # Should be (8, 12, 1)

    # Get kernel
    kernel_fn = get_3d_clifford_kernel
    res = kernel_fn(w, sig.g)
    kernel = res[1] if isinstance(res, tuple) else res
    print(f"  Kernel from kernel_fn: {type(kernel)}, shape={kernel.shape if hasattr(kernel, 'shape') else 'unknown'}")

    # View kernel
    x_reshaped = x_flat.view(B_manual * N_manual, I_x)
    print(f"  x_reshaped={x_reshaped.shape}")  # Should be (12, 8)

    try:
        kernel_view = kernel.view(B_manual * N_manual, I_x, I_x)
        print(f"  kernel.view({B_manual * N_manual}, {I_x}, {I_x}): {kernel_view.shape}")
    except Exception as e:
        print(f"  ERROR in kernel.view(): {e}")
        print(f"  kernel element type: {type(kernel)}")
        print(f"  kernel.numel(): {kernel.numel() if hasattr(kernel, 'numel') else 'N/A'}")
        print(f"  Expected numel: {B_manual * N_manual * I_x * I_x} (= {B_manual} * {N_manual} * {I_x} * {I_x})")

    # Try batched matrix multiply
    try:
        out = torch.bmm(kernel_view, x_reshaped.unsqueeze(-1)).squeeze(-1)
        print(f"  After bmm: {out.shape}")
        out_final = out.view(B_manual, N_manual, I_x)
        print(f"  Final output: {out_final.shape}")
        print(f"  Output[0, :, 0] (should all be 6.0): {out_final[0, :, 0]}")
    except Exception as e:
        print(f"  ERROR in bmm(): {e}")

    # Now call the actual function
    print(f"\nActual function result:")
    result = geometric_product(x, y, sig_g)
    print(f"  Result shape: {result.shape}")
    print(f"  Result[0, :, 0] (should all be 6.0): {result[0, :, 0]}")
    print(f"  Full result[0]: {result[0]}")


if __name__ == "__main__":
    debug_geometric_product()
