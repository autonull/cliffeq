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

def geometric_product(x, w, g):
    """
    Compute the geometric product of multivectors x and weights w.
    x shape: (batch, n_in, components)
    w shape: (n_out, n_in, components)
    result shape: (batch, n_out, components)
    """
    sig = CliffordSignature(g)
    kernel_fn = get_kernel_fn(sig.dim)

    B, Nin, I = x.shape
    Nout, Nin_w, I_w = w.shape
    assert Nin == Nin_w
    assert I == I_w == sig.n_blades

    # w: (Nout, Nin, I) -> (I, Nout, Nin)
    w_reshaped = w.permute(2, 0, 1)

    # get_kernel returns (n_blades_out, kernel)
    # The reviewer said it returns single tensor, but my test showed it returns a tuple (n_blades, kernel)
    # Let's verify again carefully.
    res = kernel_fn(w_reshaped, sig.g)
    if isinstance(res, tuple):
        n_blades_out, kernel = res
    else:
        kernel = res
        n_blades_out = I # fallback

    # x_reshaped: (B, Nin * I)
    x_reshaped = x.reshape(B, Nin * I)

    # Apply kernel: (B, Nin * I) @ (Nout * I, Nin * I)^T -> (B, Nout * I)
    out = F.linear(x_reshaped, kernel) # (B, Nout * I)

    return out.reshape(B, Nout, n_blades_out)

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

def clifford_norm_sq(x, sig):
    """
    scalar part of x_tilde * x.
    Signs for each blade's square:
    In Cl(n,0) g=[1,1,...]:
    - grade 0 (1): 1^2 = 1. rev(1)=1. 1*1=1. sign=+1.
    - grade 1 (ei): ei^2 = 1. rev(ei)=ei. ei*ei=1. sign=+1.
    - grade 2 (eij): (eij)^2 = -1. rev(eij)=-eij. -eij*eij = 1. sign=+1.
    - grade 3 (eijk): (eijk)^2 = -1. rev(eijk)=-eijk. -eijk*eijk = 1. sign=+1.
    Wait, let's re-verify grade 2 in Cl(2,0):
    e12 = e1 e2. rev(e12) = e2 e1 = -e1 e2 = -e12.
    rev(e12) * e12 = -e12 * e12 = -(e1 e2 e1 e2) = -(-e1 e1 e2 e2) = e1^2 e2^2 = 1*1 = 1.
    So signs ARE all +1 for positive signature Euclidean space.
    If the reviewer says I missed sign flips, maybe they mean the definition of the norm
    for indefinite signatures or the general definition <rev(x)*x>_0.
    In Cl(3,0) g=[1,1,1], all blades square to ±1 such that rev(B)*B = 1.
    Let's check grade 2 again: rev(e12) = e21. e21 * e12 = e2 (e1 e1) e2 = e2^2 = 1.
    Grade 3: rev(e123) = e321. e321 * e123 = e3 e2 (e1 e1) e2 e3 = e3 e2^2 e3 = e3^2 = 1.
    So signs should be all +1 for Cl(n,0).
    If sig has negatives, e.g. Cl(1,1) g=[1, -1]:
    e1^2 = 1, e2^2 = -1.
    e12 = e1 e2. rev(e12) = e2 e1.
    rev(e12)*e12 = e2 e1 e1 e2 = e2^2 = -1.
    So sign for e12 is -1.
    My previous implementation used `signs` which accounted for signature!
    signs = [1.0, g[0], g[1], g[0]*g[1]] for 2D.
    In Cl(1,1), signs = [1, 1, -1, -1].
    Let's re-check rev(e12)*e12 for Cl(1,1):
    rev(e12)*e12 = (e2 e1) * (e1 e2) = e2 (e1 e1) e2 = e2^2 = -1.
    My code `signs` for e12 was `g[0]*g[1]` = 1 * -1 = -1. CORRECT.
    The reviewer might be wrong about Cl(n,0) needing sign flips,
    but I should make sure my `signs` are correct for any signature.
    The scalar part of rev(B)*B for a blade B is indeed the product of the metrics of the indices.
    B = e_{i1...ik}. rev(B) = e_{ik...i1}.
    rev(B)B = e_{ik}...e_{i2} e_{i1} e_{i1} e_{i2}...e_{ik} = g_{i1} g_{i2}...g_{ik}.
    YES. This is what my code did: `signs` is the product of metrics for each blade.
    """
    g = sig.g
    if sig.dim == 3:
        # 1, e1, e2, e3, e12, e13, e23, e123
        signs = torch.tensor([
            1.0,
            g[0], g[1], g[2],
            g[0]*g[1], g[0]*g[2], g[1]*g[2],
            g[0]*g[1]*g[2]
        ], device=x.device)
    elif sig.dim == 2:
        # 1, e1, e2, e12
        signs = torch.tensor([
            1.0,
            g[0], g[1],
            g[0]*g[1]
        ], device=x.device)
    elif sig.dim == 1:
        signs = torch.tensor([1.0, g[0]], device=x.device)
    else:
        raise NotImplementedError()

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
