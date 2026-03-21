import torch
from cliffeq.algebra.utils import geometric_product, clifford_norm_sq, reverse, grade_project
from cliffordlayers.signature import CliffordSignature

class DynamicsRule:
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        raise NotImplementedError()

class LinearDot(DynamicsRule):
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            E = energy_fn(x_in).sum()
            grad = torch.autograd.grad(E, x_in)[0]
        return (x_in - alpha * grad).detach()

class GeomProduct(DynamicsRule):
    def __init__(self, g, normalize=False):
        self.g = g
        self.normalize = normalize
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            E = energy_fn(x_in).sum()
            grad = torch.autograd.grad(E, x_in)[0]

        gp = geometric_product(grad, x_in, self.g)

        if self.normalize:
            grad_flat = grad.reshape(grad.shape[0], -1)
            gp_flat = gp.reshape(gp.shape[0], -1)
            grad_norm = torch.norm(grad_flat, dim=-1)
            gp_norm = torch.norm(gp_flat, dim=-1)
            ratio = (grad_norm / (gp_norm + 1e-8))
            view_shape = [gp.shape[0]] + [1] * (gp.ndim - 1)
            update = gp * ratio.view(*view_shape)
            # Use smaller alpha for geometric updates if they are unstable
            return (x_in - (alpha * 0.1) * update).detach()
        return (x_in - alpha * gp).detach()

class ExpMap(DynamicsRule):
    def __init__(self, g):
        self.g = g
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        # Placeholder: exponentiating a multivector grad is non-trivial.
        # Use normalized geometric product as a stable approximation for now.
        return GeomProduct(self.g, normalize=True).step(x, energy_fn, alpha)

class RotorOnly(DynamicsRule):
    def __init__(self, g):
        self.g = g
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            E = energy_fn(x_in).sum()
            grad = torch.autograd.grad(E, x_in)[0]

        sig = CliffordSignature(self.g)
        if sig.dim == 3:
            even_mask = torch.tensor([1, 0, 0, 0, 1, 1, 1, 0], device=x.device)
        elif sig.dim == 2:
            even_mask = torch.tensor([1, 0, 0, 1], device=x.device)
        else:
            even_mask = torch.tensor([1, 0], device=x.device)

        even_x = x_in * even_mask
        even_grad = grad * even_mask

        even_update = geometric_product(even_grad, even_x, self.g)
        # Normalize even update
        even_up_norm = torch.norm(even_update.reshape(x.shape[0], -1), dim=-1)
        even_grad_norm = torch.norm(even_grad.reshape(x.shape[0], -1), dim=-1)
        ratio = (even_grad_norm / (even_up_norm + 1e-8)).view(-1, *([1]*(x.ndim-1)))
        even_update = even_update * ratio

        odd_mask = 1.0 - even_mask
        odd_update = grad * odd_mask

        update = even_update * 0.1 + odd_update # scale even part
        return (x_in - alpha * update).detach()

class Riemannian(DynamicsRule):
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            E = energy_fn(x_in).sum()
            grad = torch.autograd.grad(E, x_in)[0]
        dims = tuple(range(1, x_in.ndim))
        norm_sq = torch.sum(x_in * x_in, dim=dims, keepdim=True)
        dot = torch.sum(grad * x_in, dim=dims, keepdim=True)
        proj_grad = grad - dot * x_in / (norm_sq + 1e-8)
        x_new = x_in - alpha * proj_grad
        norm = torch.sqrt(torch.sum(x_new * x_new, dim=dims, keepdim=True))
        x_new = x_new / (norm + 1e-8)
        return x_new.detach()

class GradeSplit(DynamicsRule):
    def __init__(self, alphas_per_grade):
        self.alphas = alphas_per_grade
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        sig = CliffordSignature(energy_fn.g if hasattr(energy_fn, 'g') else torch.tensor([1.0, 1.0, 1.0]))
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            E = energy_fn(x_in).sum()
            grad = torch.autograd.grad(E, x_in)[0]
        if sig.dim == 3:
            blade_grades = [0, 1, 1, 1, 2, 2, 2, 3]
        elif sig.dim == 2:
            blade_grades = [0, 1, 1, 2]
        else:
            blade_grades = [0, 1]
        update = torch.zeros_like(grad)
        for i, grade in enumerate(blade_grades):
            a = self.alphas.get(grade, alpha)
            update[..., i] = a * grad[..., i]
        return (x_in - update).detach()

class WedgeUpdate(DynamicsRule):
    def __init__(self, g):
        self.g = g
    def step(self, x, energy_fn, alpha) -> torch.Tensor:
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            E = energy_fn(x_in).sum()
            grad = torch.autograd.grad(E, x_in)[0]

        # Wedge product x ∧ ∇E
        # Approximate with grade projection of geom product (keeping higher grades)
        sig = CliffordSignature(self.g)
        gp = geometric_product(x_in, grad, self.g)

        # Keep vector, bivector, trivector parts for the update
        update = grade_project(gp, [1, 2, 3], sig)

        # Normalize update to match gradient scale for stability
        up_norm = torch.norm(update.reshape(x.shape[0], -1), dim=-1)
        grad_norm = torch.norm(grad.reshape(x.shape[0], -1), dim=-1)
        ratio = (grad_norm / (up_norm + 1e-8)).view(-1, *([1]*(x.ndim-1)))

        return (x_in - alpha * update * ratio).detach()
