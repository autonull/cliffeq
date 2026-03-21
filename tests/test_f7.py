import torch
from cliffeq.benchmarks.metrics import equivariance_violation

def test_f7_equivariance():
    def model(x):
        return x
    def transform(x):
        # x is (B, 1, 2)
        return torch.stack([-x[..., 1], x[..., 0]], dim=-1)
    x = torch.randn(1, 1, 2)
    # Output of model is scalar-like in P1.1 (1D)
    # But model(x) here returns same shape.
    # For a general test:
    def model_inv(x):
        return torch.norm(x, dim=-1, keepdim=True)
    violation = equivariance_violation(model_inv, x, transform, lambda y: y)
    assert abs(violation) < 1e-6
    print("test_f7_equivariance passed")

if __name__ == "__main__":
    test_f7_equivariance()
