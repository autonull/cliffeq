import torch
from torch import nn
import torch.nn.functional as F
from cliffeq.algebra.utils import geometric_product, scalar_part, bivector_part, reverse
from cliffordlayers.signature import CliffordSignature

class CliffordAttention(nn.Module):
    def __init__(self, n_heads: int, clifford_dim: int, sig_g: torch.Tensor, use_orientation_bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.clifford_dim = clifford_dim
        self.register_buffer("sig_g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.use_orientation_bias = use_orientation_bias
        self.n_blades = self.sig.n_blades
        self.batch_first = True

        d_model = n_heads * clifford_dim * self.n_blades
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        if use_orientation_bias:
            n_biv = 3 if self.sig.dim == 3 else (1 if self.sig.dim == 2 else 0)
            if n_biv > 0:
                self.bias_weight = nn.Parameter(torch.randn(n_heads, n_biv) * 0.01)
            else:
                self.use_orientation_bias = False

    def get_kernel(self, device):
        from cliffeq.algebra.utils import get_kernel_fn
        kernel_fn = get_kernel_fn(self.sig.dim)
        # kernel_fn expects weight tensor of shape (I, Nout, Nin)
        # Create identity: (I, 1, 1) eye matrix for each blade
        w = torch.eye(self.n_blades, device=device).unsqueeze(1).unsqueeze(2)  # (I, 1, 1)
        res = kernel_fn(w, self.sig_g.to(device))
        return res[1] if isinstance(res, tuple) else res

    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None,
                attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None,
                need_weights: bool = True) -> tuple:
        """
        Drop-in replacement for nn.MultiheadAttention.forward.
        query, key, value: (B, L, d_model) or (L, B, d_model) if batch_first=False (not supported here)
        """
        if key is None: key = query
        if value is None: value = query

        B, L, _ = query.shape
        M = key.shape[1]

        q = self.q_proj(query).view(B, L, self.n_heads, self.clifford_dim, self.n_blades)
        k = self.k_proj(key).view(B, M, self.n_heads, self.clifford_dim, self.n_blades)
        v = self.v_proj(value).view(B, M, self.n_heads, self.clifford_dim, self.n_blades)

        q = q.transpose(1, 2) # (B, H, L, D, I)
        k = k.transpose(1, 2) # (B, H, M, D, I)
        v = v.transpose(1, 2) # (B, H, M, D, I)

        from cliffeq.algebra.utils import get_blade_signs
        signs = get_blade_signs(self.sig, query.device)

        q_rev = reverse(q, self.sig)
        q_w = q_rev * signs
        # (B, H, L, D, I), (B, H, M, D, I) -> (B, H, L, M)
        logits = torch.einsum("bhldi,bhmdi->bhlm", q_w, k) / (self.clifford_dim * self.n_blades)**0.5

        if attn_mask is not None:
            # attn_mask: (L, M) or (B*H, L, M)
            if attn_mask.dim() == 2:
                logits = logits + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                logits = logits + attn_mask.view(B, self.n_heads, L, M)

        if self.use_orientation_bias:
            kernel = self.get_kernel(query.device)
            if self.sig.dim == 3:
                biv_idx = [4, 5, 6]
            elif self.sig.dim == 2:
                biv_idx = [3]
            else:
                biv_idx = []

            if biv_idx:
                q_rev = reverse(q, self.sig)
                biv_kernel = kernel[:, :, biv_idx]  # (I, I, n_biv)
                # (B, H, L, D, I), (B, H, M, D, I), (I, I, n_biv) -> (B, H, L, M, n_biv)
                # Note: biv_kernel needs to be transposed to match the einsum contraction
                biv_part = torch.einsum("bhldi,bhmdi,ijc->bhlmc", q_rev, k, biv_kernel)
                bias = torch.einsum("bhlmc,hc->bhlm", biv_part, self.bias_weight)
                logits = logits + bias

        attn = F.softmax(logits, dim=-1)
        out = torch.einsum("bhlm,bhmdi->bhldi", attn, v)
        out = out.transpose(1, 2).reshape(B, L, -1)

        if need_weights:
            # return average weights across heads
            return out, attn.mean(1)
        return out, None
