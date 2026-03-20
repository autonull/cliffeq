import torch
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.cliffordkernels import (
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_3d_clifford_kernel,
)
import torch.nn.functional as F

def get_kernel_fn(dim):
    if dim == 1:
        return get_1d_clifford_kernel
    elif dim == 2:
        return get_2d_clifford_kernel
    elif dim == 3:
        return get_3d_clifford_kernel
    else:
        raise NotImplementedError(f"Dimension {dim} not supported")

def geometric_product(x, y, g):
    """
    Compute the geometric product of multivectors x and y.
    Supports:
    - Elementwise: x (B, N, I), y (B, N, I) -> (B, N, I)
    - Weight-based: x (B, Nin, I), y (Nout, Nin, I) -> (B, Nout, I)
    """
    sig = CliffordSignature(g)
    kernel_fn = get_kernel_fn(sig.dim)
    I = sig.n_blades

    if x.shape == y.shape:
        # Elementwise geometric product
        B, N, I_x = x.shape
        assert I_x == I

        # Flatten B and N to treat them as channels for the kernel
        x_flat = x.view(B * N, 1, I) # (BN, 1, I)
        y_flat = y.view(B * N, 1, I) # (BN, 1, I)

        # y_flat as weights: (I, BN, 1)
        w = y_flat.permute(2, 0, 1)
        res = kernel_fn(w, sig.g)
        # Fix: cliffordlayers kernel functions return (n_blades_out, weight_kernel)
        kernel = res[1] if isinstance(res, tuple) else res

        # x_flat: (BN, I)
        x_reshaped = x_flat.view(B * N, I)

        # Reshape kernel to (BN, I, I) for batched matrix multiplication
        kernel = kernel.view(B * N, I, I)
        out = torch.bmm(kernel, x_reshaped.unsqueeze(-1)).squeeze(-1)
        return out.view(B, N, I)

    elif y.ndim == 3 and x.ndim == 3:
        # Weight-based (Linear layer style)
        # x: (B, Nin, I)
        # y: (Nout, Nin, I)
        B, Nin, I_x = x.shape
        Nout, Nin_y, I_y = y.shape
        assert Nin == Nin_y and I_x == I and I_y == I

        w = y.permute(2, 0, 1)
        res = kernel_fn(w, sig.g)
        kernel = res[1] if isinstance(res, tuple) else res

        x_reshaped = x.reshape(B, Nin * I)
        out = F.linear(x_reshaped, kernel)
        return out.view(B, Nout, I)

    else:
        raise NotImplementedError(f"Geometric product for shapes {x.shape} and {y.shape} not implemented")

def grade_project(x, grades, sig):
    mask = torch.zeros(sig.n_blades, device=x.device)
    if sig.dim == 3:
        blade_grades = [0, 1, 1, 1, 2, 2, 2, 3]
    elif sig.dim == 2:
        blade_grades = [0, 1, 1, 2]
    elif sig.dim == 1:
        blade_grades = [0, 1]
    else:
        raise NotImplementedError()
    for i, g in enumerate(blade_grades):
        if g in grades:
            mask[i] = 1.0
    return x * mask

def reverse(x, sig):
    mask = torch.ones(sig.n_blades, device=x.device)
    if sig.dim == 3:
        blade_grades = [0, 1, 1, 1, 2, 2, 2, 3]
    elif sig.dim == 2:
        blade_grades = [0, 1, 1, 2]
    elif sig.dim == 1:
        blade_grades = [0, 1]
    for i, g in enumerate(blade_grades):
        if g == 2 or g == 3:
            mask[i] = -1.0
    return x * mask

def scalar_part(x):
    return x[..., 0]

def vector_part(x, sig):
    if sig.dim == 3:
        return x[..., 1:4]
    elif sig.dim == 2:
        return x[..., 1:3]
    elif sig.dim == 1:
        return x[..., 1:2]

def bivector_part(x, sig):
    if sig.dim == 3:
        return x[..., 4:7]
    elif sig.dim == 2:
        return x[..., 3:4]
    else:
        return None

def get_blade_signs(sig, device):
    g = sig.g
    if sig.dim == 3:
        # 1, e1, e2, e3, e12, e13, e23, e123
        # squares: 1, g1, g2, g3, -g1g2, -g1g3, -g2g3, -g1g2g3
        # reverse(x) * x scalar part:
        # reverse(1) = 1, sq = 1
        # reverse(ei) = ei, sq = gi
        # reverse(eij) = -eij, sq = -(-gij) = gij
        # reverse(e123) = -e123, sq = -(-g123) = g123
        signs = torch.tensor([1.0, g[0], g[1], g[2], g[0]*g[1], g[0]*g[2], g[1]*g[2], g[0]*g[1]*g[2]], device=device)
    elif sig.dim == 2:
        # 1, e1, e2, e12
        # reverse(1)=1, sq=1
        # reverse(ei)=ei, sq=gi
        # reverse(e12)=-e12, sq=-(-g1g2) = g1g2
        signs = torch.tensor([1.0, g[0], g[1], g[0]*g[1]], device=device)
    elif sig.dim == 1:
        # 1, e1
        signs = torch.tensor([1.0, g[0]], device=device)
    else:
        raise NotImplementedError()
    return signs

def clifford_norm_sq(x, sig):
    signs = get_blade_signs(sig, x.device)
    return torch.sum((x ** 2) * signs, dim=-1)

def embed_scalar(x, sig):
    res = torch.zeros((*x.shape, sig.n_blades), device=x.device)
    res[..., 0] = x
    return res

def embed_vector(x, sig):
    res = torch.zeros((*x.shape[:-1], sig.n_blades), device=x.device)
    if sig.dim == 3:
        res[..., 1:4] = x
    elif sig.dim == 2:
        res[..., 1:3] = x
    elif sig.dim == 1:
        res[..., 1:2] = x
    return res

def grade_soft_threshold(x, lambdas, sig):
    """
    Apply soft thresholding independently per grade.
    lambdas: list or dict of thresholds per grade {grade: value}
    """
    if sig.dim == 3:
        blade_grades = [0, 1, 1, 1, 2, 2, 2, 3]
    elif sig.dim == 2:
        blade_grades = [0, 1, 1, 2]
    else:
        blade_grades = [0, 1]

    out = x.clone()
    for i, g in enumerate(blade_grades):
        l = lambdas[g] if isinstance(lambdas, (list, torch.Tensor)) else lambdas.get(g, 0.0)
        if l > 0:
            out[..., i] = torch.sign(x[..., i]) * torch.relu(torch.abs(x[..., i]) - l)
    return out
