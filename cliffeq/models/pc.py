import torch
from torch import nn
from cliffeq.algebra.utils import geometric_product
from cliffordlayers.signature import CliffordSignature

class CliffordPC(nn.Module):
    """
    Clifford Predictive Coding.
    Layer l predicts layer l-1.
    """
    def __init__(self, layer_dims, sig_g):
        super().__init__()
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(layer_dims[i-1], layer_dims[i], self.sig.n_blades) * 0.1)
            for i in range(1, len(layer_dims))
        ])

    def forward(self, x: torch.Tensor, n_iter=20, alpha=0.1) -> list:
        # x: input multivector (B, nodes, I)
        B = x.shape[0]
        states = [x.detach().clone()]
        for w in self.weights:
            # Initialize next layer state. w shape is (N_{l-1}, N_l, I)
            states.append(torch.zeros((B, w.shape[1], self.sig.n_blades), device=x.device))

        for _ in range(n_iter):
            for l in range(1, len(states)):
                # x_hat_{l-1} = W_l * x_l
                x_hat = geometric_product(states[l], self.weights[l-1], self.g)
                err = states[l-1] - x_hat

                # Update states[l] to minimize error
                # dE/dx_l = - (W_l)^T * err
                from cliffeq.algebra.utils import reverse
                w_rev = reverse(self.weights[l-1], self.sig)
                w_T = w_rev.transpose(0, 1)

                grad = -geometric_product(err, w_T, self.g)
                states[l] = states[l] - alpha * grad

        return states
