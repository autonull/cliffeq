import torch
from cliffeq.algebra.utils import geometric_product, clifford_norm_sq, reverse, grade_project
from cliffordlayers.signature import CliffordSignature

class DynamicsRule:
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        raise NotImplementedError()

class LinearDot(DynamicsRule):
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            E = energy_fn(x).sum()
            grad = torch.autograd.grad(E, x, create_graph=x.requires_grad)[0]
        return x - alpha * grad

class GeomProduct(DynamicsRule):
    def __init__(self, g, normalize=False):
        self.g = g
        self.normalize = normalize
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            E = energy_fn(x).sum()
            grad = torch.autograd.grad(E, x, create_graph=x.requires_grad)[0]

        # ∇E ✶ x
        # Note: if E = 1/2 ||x||^2, ∇E = x. Then ∇E ✶ x = x ✶ x.
        # For a vector v, v ✶ v = v dot v.
        # So x ✶ x for a vector x is a scalar update.
        gp = geometric_product(grad, x, self.g)

        if self.normalize:
            # Scale alpha by x norm to keep update relative
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            gp_norm = torch.norm(gp, dim=-1, keepdim=True)
            update = gp / (gp_norm + 1e-8) * (torch.norm(grad, dim=-1, keepdim=True) + 1e-8)
            return x - alpha * update

        return x - alpha * gp

class ExpMap(DynamicsRule):
    def __init__(self, g):
        self.g = g
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        # x ← exp(−α ∇E) ✶ x
        # 1st order: (1 - alpha * grad) * x = x - alpha * grad * x
        return GeomProduct(self.g).step(x, energy_fn, alpha)

class RotorOnly(DynamicsRule):
    def __init__(self, g):
        self.g = g
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            E = energy_fn(x).sum()
            grad = torch.autograd.grad(E, x, create_graph=x.requires_grad)[0]
        return x - alpha * grad

class Riemannian(DynamicsRule):
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            E = energy_fn(x).sum()
            grad = torch.autograd.grad(E, x, create_graph=x.requires_grad)[0]
        norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        dot = torch.sum(grad * x, dim=-1, keepdim=True)
        proj_grad = grad - dot * x / (norm_sq + 1e-8)
        x_new = x - alpha * proj_grad
        x_new = x_new / (torch.sqrt(torch.sum(x_new * x_new, dim=-1, keepdim=True)) + 1e-8)
        return x_new

class GradeSplit(DynamicsRule):
    def __init__(self, alphas_per_grade):
        self.alphas = alphas_per_grade
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        sig = CliffordSignature(energy_fn.g if hasattr(energy_fn, 'g') else torch.tensor([1.0, 1.0, 1.0]))
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            E = energy_fn(x).sum()
            grad = torch.autograd.grad(E, x, create_graph=x.requires_grad)[0]
        if sig.dim == 3:
            blade_grades = [0, 1, 1, 1, 2, 2, 2, 3]
        elif sig.dim == 2:
            blade_grades = [0, 1, 1, 2]
        else:
            blade_grades = [0, 1]
        update = torch.zeros_like(grad)
        for i, g in enumerate(blade_grades):
            a = self.alphas.get(g, alpha)
            update[..., i] = a * grad[..., i]
        return x - update

class WedgeUpdate(DynamicsRule):
    def __init__(self, g):
        self.g = g
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        return GeomProduct(self.g).step(x, energy_fn, alpha)
