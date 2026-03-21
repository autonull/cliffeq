import torch
from torch import nn
import torch.nn.functional as F
from cliffeq.algebra.utils import scalar_part
from cliffordlayers.signature import CliffordSignature

class CliffordPC(nn.Module):
    """
    Clifford Predictive Coding (Simplified).
    Layer l predicts layer l-1 using learned linear transformations on scalar parts.
    """
    def __init__(self, layer_dims, sig_g):
        super().__init__()
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.layer_dims = layer_dims

        # Scalar prediction weights (simplified: work on scalar parts)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(layer_dims[i-1], layer_dims[i]) * 0.1)
            for i in range(1, len(layer_dims))
        ])

    def forward(self, x: torch.Tensor, n_iter=20, alpha=0.1) -> list:
        # x: input multivector (B, nodes, I)
        B = x.shape[0]
        states = [x.detach().clone()]

        # Initialize hidden layer states from input's scalar part
        x_scalar = scalar_part(x)  # (B, nodes)
        for i in range(1, len(self.layer_dims)):
            h = torch.randn(B, self.layer_dims[i], self.sig.n_blades, device=x.device) * 0.1
            states.append(h)

        # PC iterations
        for it in range(n_iter):
            for l in range(1, len(states)):
                # Predict layer l-1's scalar part from layer l's scalar part
                state_l_scalar = scalar_part(states[l])  # (B, layer_dims[l])
                x_hat_scalar = F.linear(state_l_scalar, self.weights[l-1])  # (B, layer_dims[l-1])

                # Error in scalar space
                state_l_minus_1_scalar = scalar_part(states[l-1])  # (B, layer_dims[l-1])
                err_scalar = state_l_minus_1_scalar - x_hat_scalar

                # Update states[l] to minimize error (no in-place ops for gradient tracking)
                new_state = states[l].clone()
                new_state[..., 0] = new_state[..., 0] - alpha * err_scalar.mean(dim=1, keepdim=True)
                states[l] = new_state

        return states
