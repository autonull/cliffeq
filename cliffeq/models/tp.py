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
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(layer_dims[i], layer_dims[i-1], self.sig.n_blades) * 0.1)
            for i in range(1, len(layer_dims))
        ])

    def forward(self, x: torch.Tensor) -> list:
        # Standard forward pass
        activations = [x]
        for w in self.weights:
            # y = W * x
            # x: (B, Nin, I), W: (Nout, Nin, I) -> (B, Nout, I)
            y = geometric_product(activations[-1], w, self.g)
            activations.append(y)
        return activations

    def compute_targets(self, activations, global_target):
        # activations: list of [x_0, x_1, ..., x_L]
        targets = [None] * len(activations)
        targets[-1] = global_target

        for l in range(len(self.weights) - 1, -1, -1):
            # activations[l+1] = f(activations[l])
            # target[l] \approx f^-1(target[l+1])
            w = self.weights[l]
            w_rev = reverse(w, self.sig)

            # Approximation of inverse: W^-1 \approx W_rev / ||W||^2
            # Here ||W||^2 can be simplified as scalar part of (W_rev * W)
            # For simplicity, we just use the reversal as an approximation
            # f(x) = geometric_product(x, W, g)
            # So x \approx geometric_product(y, W_rev_T, g) / norm
            w_rev_T = w_rev.transpose(0, 1) # (Nin, Nout, I)

            target_next = targets[l+1]
            # y: (B, Nout, I), w_rev_T: (Nin, Nout, I) -> (B, Nin, I)
            target_prev = geometric_product(target_next, w_rev_T, self.g)

            # Normalize by average squared norm of weights
            n2 = clifford_norm_sq(w, self.sig).mean()
            targets[l] = target_prev / (n2 + 1e-8)

        return targets
