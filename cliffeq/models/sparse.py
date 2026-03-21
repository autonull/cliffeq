import torch
from torch import nn
from cliffeq.algebra.utils import geometric_product, reverse, get_blade_signs, grade_soft_threshold
from cliffordlayers.signature import CliffordSignature

class CliffordISTA(nn.Module):
    """
    Clifford-valued Iterative Soft-Thresholding Algorithm (ISTA).
    Minimizes: 0.5 * ||y - A x||^2 + Σ_k λ_k ||<x>_k||_1
    """
    def __init__(self, A, sig_g, lambdas, n_iter=50, step_size=0.01):
        super().__init__()
        self.A = nn.Parameter(A) # (Nout, Nin, I)
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.lambdas = lambdas # list/dict of λ per grade
        self.n_iter = n_iter
        self.step_size = step_size

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, Nout, I)
        B = y.shape[0]
        Nin = self.A.shape[1]
        x = torch.zeros((B, Nin, self.sig.n_blades), device=y.device)

        # Precompute signs for scalar part logic if needed
        # We use geometric_product for A*x

        for _ in range(self.n_iter):
            # grad = A^T (Ax - y)
            Ax = geometric_product(x, self.A, self.g)
            err = Ax - y

            # A^T is reverse(A) in Clifford space?
            # Actually, for matrix A, adjoint is reverse(A).T
            A_rev = reverse(self.A, self.sig)
            # geometric_product(err, A_rev.transpose(0, 1))
            # x: (B, Nin, I), A: (Nout, Nin, I)
            # err: (B, Nout, I)
            # grad should be (B, Nin, I)
            # W: (hidden, in, I) in BilinearEnergy. W is A here.
            # geometric_product(err, A_rev)? Need to check shape.
            # A_rev is (Nout, Nin, I). Transpose to (Nin, Nout, I)
            A_T = A_rev.transpose(0, 1)
            grad = geometric_product(err, A_T, self.g)

            x = x - self.step_size * grad
            x = grade_soft_threshold(x, self.lambdas, self.sig)

        return x

class CliffordLISTA(nn.Module):
    """
    Learned ISTA (LISTA) variant with Clifford states.
    """
    def __init__(self, in_nodes, hidden_nodes, sig_g, n_layers=5):
        super().__init__()
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.n_layers = n_layers

        self.W1 = nn.Parameter(torch.randn(hidden_nodes, in_nodes, self.sig.n_blades) * 0.1)
        self.W2 = nn.Parameter(torch.randn(hidden_nodes, hidden_nodes, self.sig.n_blades) * 0.1)
        self.lambdas = nn.Parameter(torch.ones(self.sig.dim + 1) * 0.01)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, in_nodes, I)
        B = y.shape[0]
        x = torch.zeros((B, self.W1.shape[0], self.sig.n_blades), device=y.device)

        for _ in range(self.n_layers):
            # x = soft_thresh(W1 y + W2 x, lambda)
            # This is one variation of LISTA
            W1y = geometric_product(y, self.W1, self.g)
            W2x = geometric_product(x, self.W2, self.g)
            x = grade_soft_threshold(W1y + W2x, self.lambdas, self.sig)

        return x
