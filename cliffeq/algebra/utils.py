import torch
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.cliffordkernels import (
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_3d_clifford_kernel,
)
import torch.nn.functional as F
import os

_CL41_TABLE = None

def get_cl41_table(device):
    global _CL41_TABLE
    if _CL41_TABLE is None:
        path = os.path.join(os.path.dirname(__file__), "cl41_table.pt")
        if os.path.exists(path):
            _CL41_TABLE = torch.load(path, map_location=device)
        else:
            raise FileNotFoundError(f"CGA table not found at {path}. Run generate_cga_table.py first.")
    return _CL41_TABLE.to(device)

def get_kernel_fn(dim):
    if dim == 1:
        return get_1d_clifford_kernel
    elif dim == 2:
        return get_2d_clifford_kernel
    elif dim == 3:
        return get_3d_clifford_kernel
    else:
        raise NotImplementedError(f"Dimension {dim} not supported by cliffordlayers kernels.")

def geometric_product(x, y, g):
    """
    Compute the geometric product of multivectors x and y.
    Supports:
    - Elementwise: x (B, N, I), y (B, N, I) -> (B, N, I)
    - Weight-based: x (B, Nin, I), y (Nout, Nin, I) -> (B, Nout, I)

    Weight-based case is detected when y.shape[0] != x.shape[0] (different first dimension),
    which indicates y is a weight matrix, not a batch of activations.
    """
    dim = len(g)
    if dim == 5:
        # Special case for CGA since CliffordSignature might not support 5D
        I = 32
        # Precomputed table based product
        table = get_cl41_table(x.device) # (32, 32, 32)
        # table[i, j, k] is the coefficient of blade k in the product of blade i and blade j

        # Elementwise case (B, N, 32) * (B, N, 32) -> (B, N, 32)
        if x.shape == y.shape:
            # res_k = sum_i sum_j x_i * y_j * table[i, j, k]
            # Use einsum for efficiency
            return torch.einsum("bni,bnj,ijk->bnk", x, y, table)

        # Weight-based case (B, Nin, 32) * (Nout, Nin, 32) -> (B, Nout, 32)
        is_weight_based = (y.ndim == 3 and x.ndim == 3 and
                           y.shape[0] != x.shape[0] and
                           y.shape[1] == x.shape[1])
        if is_weight_based:
            # We want (B, Nout, 32)
            # res_{b, o, k} = sum_{n} sum_i sum_j x_{b, n, i} * y_{o, n, j} * table[i, j, k]
            return torch.einsum("bni,onj,ijk->bok", x, y, table)

        raise NotImplementedError(f"CGA product not implemented for shapes {x.shape} and {y.shape}")

    sig = CliffordSignature(g)
    kernel_fn = get_kernel_fn(sig.dim)
    I = sig.n_blades


    # Check if this is weight-based (y is a weight matrix, not batched activations)
    # Weight matrices have shape (Nout, Nin, I) while batched activations have shape (B, N, I)
    # The key difference: y.shape[0] != x.shape[0] for weights, and y.shape != x.shape
    is_weight_based = (y.ndim == 3 and x.ndim == 3 and
                       y.shape[0] != x.shape[0] and  # Different first dimension
                       y.shape[1] == x.shape[1])      # Same middle dimension (Nin)

    if is_weight_based:
        # Weight-based (Linear layer style)
        # x: (B, Nin, I)
        # y: (Nout, Nin, I)
        B, Nin, I_x = x.shape
        Nout, Nin_y, I_y = y.shape
        assert Nin == Nin_y and I_x == I and I_y == I, f"Shape mismatch: x={x.shape}, y={y.shape}, I={I}, expected y=(Nout, {Nin}, {I})"

        w = y.permute(2, 0, 1)
        res = kernel_fn(w, sig.g)
        kernel = res[1] if isinstance(res, tuple) else res

        x_reshaped = x.reshape(B, Nin * I)
        out = F.linear(x_reshaped, kernel)
        result = out.view(B, Nout, I)
        #print(f"    DEBUG: geometric_product weight-based: x={x.shape} w={y.shape} -> {result.shape}")
        return result

    elif x.shape == y.shape:
        # Elementwise geometric product (B, N, I) * (B, N, I) -> (B, N, I)
        B, N, I_x = x.shape
        assert I_x == I
        BN = B * N

        # Reshape x and y for vectorized kernel application
        # W must be (I, Nout, Nin) = (I, BN, 1)
        W = y.reshape(BN, I).t().unsqueeze(-1)
        res = kernel_fn(W, sig.g)
        kernel_all = res[1] if isinstance(res, tuple) else res
        # kernel_all is (BN*I, I), needs reshape to (BN, I, I)
        # Verified that .view(I, BN, I).permute(1, 0, 2) is correct
        kernels = kernel_all.reshape(I, BN, I).permute(1, 0, 2)

        x_flat = x.reshape(BN, I)
        out_flat = torch.einsum("bij,bj->bi", kernels, x_flat)
        return out_flat.reshape(B, N, I)

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
    if sig is None: # CGA case
        res = torch.zeros((*x.shape[:-1], 32), device=x.device)
        res[..., 1:4] = x[..., :3] # standard 3D vector embedding into CGA
        return res
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
