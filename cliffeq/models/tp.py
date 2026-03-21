import torch
from torch import nn
from cliffeq.algebra.utils import geometric_product, reverse, clifford_norm_sq
from cliffordlayers.signature import CliffordSignature

class CliffordTP(nn.Module):
    """
    Clifford Target Propagation.
    Uses geometric inversion via reversal for layer targets.
    f(x) = W * x
    f^-1(y) \approx W̃ * y / ||W||^2
    """
    def __init__(self, layer_dims, sig_g):
        super().__init__()
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.layer_dims = layer_dims
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(layer_dims[i], layer_dims[i-1], self.sig.n_blades) * 0.1)
            for i in range(1, len(layer_dims))
        ])

    def forward(self, x: torch.Tensor) -> list:
        # Standard forward pass - using weight-based geometric product explicitly
        activations = [x]
        for i, w in enumerate(self.weights):
            # Apply weight-based geometric product: (B, Nin, I) @ (Nout, Nin, I) -> (B, Nout, I)
            x_prev = activations[-1]
            B, Nin, I = x_prev.shape
            Nout, Nin_w, I_w = w.shape

            assert Nin == Nin_w and I == I_w, f"Layer {i}: shape mismatch x={x_prev.shape}, w={w.shape}"

            # Apply weight-based geometric product directly
            w_perm = w.permute(2, 0, 1)  # (I, Nout, Nin)
            from cliffordlayers.cliffordkernels import get_1d_clifford_kernel, get_2d_clifford_kernel, get_3d_clifford_kernel

            kernel_fn = [get_1d_clifford_kernel, get_2d_clifford_kernel, get_3d_clifford_kernel][self.sig.dim - 1]
            res = kernel_fn(w_perm, self.g)
            kernel = res[1] if isinstance(res, tuple) else res

            x_reshaped = x_prev.reshape(B, Nin * I)
            import torch.nn.functional as F
            out = F.linear(x_reshaped, kernel)
            y = out.view(B, Nout, I)

            activations.append(y)
        return activations

    def compute_targets(self, activations, global_target):
        # activations: list of [x_0, x_1, ..., x_L]
        targets = [None] * len(activations)
        targets[-1] = global_target

        from cliffordlayers.cliffordkernels import get_1d_clifford_kernel, get_2d_clifford_kernel, get_3d_clifford_kernel
        import torch.nn.functional as F

        kernel_fn = [get_1d_clifford_kernel, get_2d_clifford_kernel, get_3d_clifford_kernel][self.sig.dim - 1]

        for l in range(len(self.weights) - 1, -1, -1):
            # activations[l+1] = f(activations[l])
            # target[l] \approx f^-1(target[l+1])
            w = self.weights[l]
            w_rev = reverse(w, self.sig)
            w_rev_T = w_rev.transpose(0, 1) # (Nin_l, Nout_l, I)

            target_next = targets[l+1]  # (B, Nout_l, I)
            B, Nout_l, I = target_next.shape
            Nin_l, Nout_l_check, I_check = w_rev_T.shape

            assert Nout_l == Nout_l_check and I == I_check, f"Layer {l}: target-invert mismatch"

            # Apply weight-based geometric product for the inverse: (B, Nout_l, I) @ (Nin_l, Nout_l, I) -> (B, Nin_l, I)
            w_rev_T_perm = w_rev_T.permute(2, 0, 1)  # (I, Nin_l, Nout_l)
            res = kernel_fn(w_rev_T_perm, self.g)
            kernel = res[1] if isinstance(res, tuple) else res

            target_reshaped = target_next.reshape(B, Nout_l * I)
            out = F.linear(target_reshaped, kernel)
            target_prev = out.view(B, Nin_l, I)

            # Normalize by average squared norm of weights
            n2 = clifford_norm_sq(w, self.sig).mean()
            targets[l] = target_prev / (n2 + 1e-8)

        return targets
