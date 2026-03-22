import torch
from torch import nn
import torch.nn.functional as F
from cliffeq.algebra.utils import geometric_product, reverse, scalar_part
from cliffordlayers.signature import CliffordSignature

class CliffordPC(nn.Module):
    """
    Clifford Predictive Coding.
    Layer l predicts layer l-1: x̂_{l-1} = W_l * x_l
    Iterative inference finds x_l that minimizes prediction errors.
    """
    def __init__(self, layer_dims, sig_g):
        super().__init__()
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.layer_dims = layer_dims

        # Multivector weights: weights[l-1] maps states[l] to states[l-1]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(layer_dims[i-1], layer_dims[i], self.sig.n_blades) * 0.01)
            for i in range(1, len(layer_dims))
        ])

    def forward(self, x: torch.Tensor, n_iter=20, alpha=0.01) -> list:
        # x: input multivector (B, N_0, I)
        B = x.shape[0]
        states = [x.detach().clone()]

        # Initialize hidden layer multivector states [x_1, x_2, ..., x_L]
        for i in range(1, len(self.layer_dims)):
            h = torch.randn(B, self.layer_dims[i], self.sig.n_blades, device=x.device) * 0.001
            states.append(h)

        # PC iterations
        for it in range(n_iter):
            for l in range(1, len(states)):
                # 1. Error from prediction of layer l-1 (downward error)
                W_l = self.weights[l-1]

                pred_prev = geometric_product(states[l], W_l, self.g)
                err_prev = states[l-1] - pred_prev

                W_l_rev = reverse(W_l, self.sig)
                W_l_rev_T = W_l_rev.transpose(0, 1) # (N_l, N_l-1, I)
                grad_down = -geometric_product(err_prev, W_l_rev_T, self.g)

                grad = grad_down

                # 2. Error from prediction of layer l (upward error) - if not last layer
                if l < len(states) - 1:
                    W_next = self.weights[l]
                    pred_curr = geometric_product(states[l+1], W_next, self.g)
                    err_curr = states[l] - pred_curr
                    grad = grad + err_curr

                # Clip gradient to prevent explosion
                grad = torch.clamp(grad, -1.0, 1.0)
                states[l] = states[l] - alpha * grad

        return states

    def predict_all(self, states):
        preds = [states[0]]
        for l in range(1, len(states)):
            preds.append(geometric_product(states[l], self.weights[l-1], self.g))
        return preds
