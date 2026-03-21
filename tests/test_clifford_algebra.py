"""
Unit tests for Clifford algebra operations.
Validates correctness of geometric_product, scalar_part, reverse, etc.
"""

import torch
import pytest
from cliffeq.algebra.utils import (
    geometric_product,
    scalar_part,
    reverse,
    grade_project,
    clifford_norm_sq,
    embed_vector,
    embed_scalar
)
from cliffordlayers.signature import CliffordSignature


class TestCliffordAlgebra:
    """Test suite for Clifford algebra operations."""

    @pytest.fixture
    def sig_3d(self):
        """3D Clifford signature Cl(3,0)."""
        sig_g = torch.tensor([1.0, 1.0, 1.0])
        return CliffordSignature(sig_g), sig_g

    @pytest.fixture
    def sig_2d(self):
        """2D Clifford signature Cl(2,0)."""
        sig_g = torch.tensor([1.0, 1.0])
        return CliffordSignature(sig_g), sig_g

    def test_scalar_part_extraction(self, sig_3d):
        """Test that scalar_part correctly extracts the scalar component."""
        sig, sig_g = sig_3d
        B, N, I = 4, 3, sig.n_blades  # batch=4, nodes=3, blades=8

        # Create multivector with known scalar parts
        h = torch.randn(B, N, I)
        h[:, :, 0] = 5.0  # Set scalar part to 5.0

        scalar = scalar_part(h)

        assert scalar.shape == (B, N), f"Expected shape {(B, N)}, got {scalar.shape}"
        assert torch.allclose(scalar, torch.full((B, N), 5.0)), "Scalar part not extracted correctly"

    def test_embed_scalar(self, sig_3d):
        """Test that embed_scalar creates correct multivector."""
        sig, sig_g = sig_3d
        B, N = 4, 3

        x = torch.ones(B, N)
        h = embed_scalar(x, sig)

        assert h.shape == (B, N, sig.n_blades), f"Expected shape {(B, N, sig.n_blades)}, got {h.shape}"
        assert torch.allclose(h[..., 0], x), "Scalar component not set correctly"
        assert torch.allclose(h[..., 1:], torch.zeros(B, N, sig.n_blades - 1)), "Non-scalar components not zero"

    def test_embed_vector(self, sig_3d):
        """Test that embed_vector creates correct multivector."""
        sig, sig_g = sig_3d
        B, N = 4, 3

        x = torch.randn(B, N, 3)
        h = embed_vector(x, sig)

        assert h.shape == (B, N, sig.n_blades), f"Expected shape {(B, N, sig.n_blades)}, got {h.shape}"
        assert torch.allclose(h[..., 0], torch.zeros(B, N)), "Scalar component should be zero"
        assert torch.allclose(h[..., 1:4], x), "Vector components not set correctly"

    def test_reverse_scalar(self, sig_3d):
        """Test that reverse of scalar returns scalar (grade 0 is even)."""
        sig, sig_g = sig_3d
        B, N, I = 4, 3, sig.n_blades

        h = torch.zeros(B, N, I)
        h[..., 0] = 5.0  # Scalar

        h_rev = reverse(h, sig)

        assert torch.allclose(h_rev[..., 0], h[..., 0]), "Reverse of scalar should be scalar"

    def test_reverse_vector(self, sig_3d):
        """Test that reverse of vector returns vector (grade 1 is odd, so sign changes... wait, no)."""
        sig, sig_g = sig_3d
        B, N, I = 4, 3, sig.n_blades

        h = torch.zeros(B, N, I)
        h[..., 1:4] = torch.ones(B, N, 3)  # Vectors

        h_rev = reverse(h, sig)

        # Grade 1 elements: reverse(e_i) = e_i (still positive)
        # Reverse only negates grades 2 and 3
        assert torch.allclose(h_rev[..., 1:4], h[..., 1:4]), "Reverse of vector should be vector (same sign)"

    def test_reverse_bivector(self, sig_3d):
        """Test that reverse of bivector returns negative bivector (grade 2)."""
        sig, sig_g = sig_3d
        B, N, I = 4, 3, sig.n_blades

        h = torch.zeros(B, N, I)
        h[..., 4:7] = torch.ones(B, N, 3)  # Bivectors

        h_rev = reverse(h, sig)

        # Grade 2: reverse(e_ij) = -e_ij
        assert torch.allclose(h_rev[..., 4:7], -h[..., 4:7]), "Reverse of bivector should negate"

    def test_geometric_product_elementwise_scalar(self, sig_3d):
        """Test geometric product of two scalars = scalar product."""
        sig, sig_g = sig_3d
        B, N, I = 4, 3, sig.n_blades

        # Two scalars: (a, 0, 0, ...) * (b, 0, 0, ...) = (a*b, 0, ...)
        x = torch.zeros(B, N, I)
        y = torch.zeros(B, N, I)
        x[..., 0] = 2.0
        y[..., 0] = 3.0

        result = geometric_product(x, y, sig_g)

        assert result.shape == (B, N, I), f"Expected shape {(B, N, I)}, got {result.shape}"
        assert torch.allclose(result[..., 0], torch.full((B, N), 6.0), atol=1e-5), \
            f"Scalar product failed: expected 6.0, got {result[0, 0, 0]}"
        assert torch.allclose(result[..., 1:], torch.zeros(B, N, I - 1), atol=1e-5), \
            "Non-scalar components should be zero"

    def test_geometric_product_vector_with_self(self, sig_3d):
        """Test that v * v = ||v||^2 (scalar)."""
        sig, sig_g = sig_3d
        B, N, I = 4, 3, sig.n_blades

        # Embed vectors with known norm
        v = torch.zeros(B, N, I)
        v[..., 1:4] = torch.ones(B, N, 3)  # Unit vectors (1, 1, 1) in each component
        # ||v||^2 = 1 + 1 + 1 = 3 for Cl(3,0) with signature (+,+,+)

        result = geometric_product(v, v, sig_g)

        # v * v should give scalar part = ||v||^2 = 3
        assert result.shape == (B, N, I), f"Expected shape {(B, N, I)}, got {result.shape}"
        expected_norm_sq = 3.0
        actual_scalar = result[..., 0]
        assert torch.allclose(actual_scalar, torch.full((B, N), expected_norm_sq), atol=1e-5), \
            f"Vector self-product failed: expected {expected_norm_sq}, got {actual_scalar[0, 0]}"

    def test_geometric_product_weight_style(self, sig_3d):
        """Test weight-based geometric product (Linear layer style)."""
        sig, sig_g = sig_3d
        B, Nin, Nout, I = 4, 3, 2, sig.n_blades

        x = torch.randn(B, Nin, I)
        w = torch.randn(Nout, Nin, I)

        result = geometric_product(x, w, sig_g)

        assert result.shape == (B, Nout, I), f"Expected shape {(B, Nout, I)}, got {result.shape}"

    def test_clifford_norm_sq(self, sig_3d):
        """Test Clifford norm computation."""
        sig, sig_g = sig_3d
        B, N, I = 4, 3, sig.n_blades

        h = torch.zeros(B, N, I)
        h[..., 0] = 2.0  # Scalar

        norm_sq = clifford_norm_sq(h, sig)

        assert norm_sq.shape == (B, N), f"Expected shape {(B, N)}, got {norm_sq.shape}"
        # For Cl(3,0): scalar_sq * 1 = 4
        assert torch.allclose(norm_sq, torch.full((B, N), 4.0), atol=1e-5), \
            f"Norm_sq failed: expected 4.0, got {norm_sq[0, 0]}"

    def test_grade_project_scalar(self, sig_3d):
        """Test grade projection extracts only scalars."""
        sig, sig_g = sig_3d
        B, N, I = 4, 3, sig.n_blades

        h = torch.randn(B, N, I)
        h_scalar = grade_project(h, [0], sig)

        assert torch.allclose(h_scalar[..., 0], h[..., 0]), "Scalar component not preserved"
        assert torch.allclose(h_scalar[..., 1:], torch.zeros(B, N, I - 1)), "Non-scalar components not zero"

    def test_geometric_product_commutativity_violation(self, sig_3d):
        """Test that geometric product is NOT commutative (general property)."""
        sig, sig_g = sig_3d
        B, N, I = 2, 1, sig.n_blades

        x = torch.randn(B, N, I)
        y = torch.randn(B, N, I)

        xy = geometric_product(x, y, sig_g)
        yx = geometric_product(y, x, sig_g)

        # They should generally NOT be equal
        is_commutative = torch.allclose(xy, yx, atol=1e-5)
        assert not is_commutative, "Geometric product should not be commutative in general"


class TestGeometricProductEdgeCases:
    """Test edge cases and potential bugs in geometric_product."""

    def test_noncontiguous_tensor_handling(self):
        """Test that geometric_product handles non-contiguous tensors correctly."""
        sig_g = torch.tensor([1.0, 1.0, 1.0])
        sig = CliffordSignature(sig_g)
        I = sig.n_blades

        # Create non-contiguous tensor via transpose and permute
        x_temp = torch.randn(3, 4, I)
        x = x_temp.permute(1, 0, 2)  # Now (4, 3, I) but non-contiguous
        y = torch.randn(4, 3, I)

        assert not x.is_contiguous(), "x should be non-contiguous"

        # This should not crash (previously failed with view)
        result = geometric_product(x, y, sig_g)
        assert result.shape == (4, 3, I), f"Expected shape {(4, 3, I)}, got {result.shape}"

    def test_small_batch(self):
        """Test with batch size 1."""
        sig_g = torch.tensor([1.0, 1.0, 1.0])
        sig = CliffordSignature(sig_g)
        I = sig.n_blades

        x = torch.randn(1, 2, I)
        y = torch.randn(1, 2, I)

        result = geometric_product(x, y, sig_g)
        assert result.shape == (1, 2, I)

    def test_single_node(self):
        """Test with single node."""
        sig_g = torch.tensor([1.0, 1.0, 1.0])
        sig = CliffordSignature(sig_g)
        I = sig.n_blades

        x = torch.randn(4, 1, I)
        y = torch.randn(4, 1, I)

        result = geometric_product(x, y, sig_g)
        assert result.shape == (4, 1, I)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
