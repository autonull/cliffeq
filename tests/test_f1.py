import torch
from cliffeq.algebra.utils import geometric_product, clifford_norm_sq, reverse, embed_scalar, embed_vector
from cliffordlayers.signature import CliffordSignature

def test_embed():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    x = torch.tensor([1.0, 2.0])
    e_s = embed_scalar(x, sig)
    assert e_s.shape == (2, 4)
    assert e_s[0, 0] == 1.0
    assert e_s[1, 0] == 2.0
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    e_v = embed_vector(v, sig)
    assert e_v.shape == (2, 4)
    assert e_v[0, 1] == 1.0
    assert e_v[1, 2] == 1.0
    print("test_embed passed")

def test_norm_sq():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    x = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    n2 = clifford_norm_sq(x, sig)
    assert n2.item() == 2.0
    x = torch.tensor([[0.0, 0.0, 0.0, 1.0]]) # e12
    n2 = clifford_norm_sq(x, sig)
    # scalar part of reverse(e12) * e12 = -e12 * e12 = -(-e1*e1 * e2*e2) = 1
    # Actually reverse(e12) = -e12, e12^2 = e1 e2 e1 e2 = -e1 e1 e2 e2 = -1*1*1 = -1
    # So reverse(e12) * e12 = -(-1) = 1.
    assert n2.item() == 1.0
    print("test_norm_sq passed")

if __name__ == "__main__":
    test_embed()
    test_norm_sq()
