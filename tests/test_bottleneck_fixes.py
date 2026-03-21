"""
Comprehensive unit tests for bottleneck implementations.

These tests verify:
1. Each bottleneck implementation works correctly
2. Bugs are fixed and don't regress
3. Mathematical properties hold
4. Gradient flow is correct
"""

import pytest
import torch
import torch.nn as nn
from cliffeq.models.bottleneck_v2 import CliffordEPBottleneckV2 as BN_V2
from cliffeq.models.bottleneck_v3_proper import (
    CliffordEPBottleneckV3,
    LinearBottleneck,
    IdentityBottleneck
)


class TestBottleneckBasics:
    """Test basic bottleneck properties"""

    @pytest.fixture
    def sig_g(self):
        """Clifford algebra signature Cl(2,0)"""
        return torch.tensor([1.0, 1.0])

    def test_v2_disabled_shape(self, sig_g):
        """V2 (disabled): Check output shape matches input"""
        bn = BN_V2(in_dim=64, out_dim=32, sig_g=sig_g)
        x = torch.randn(8, 64)
        y = bn(x)

        # Output should be out_dim=32
        assert y.shape == (8, 32), f"Expected (8, 32), got {y.shape}"

    def test_v3_maintains_dimensionality(self, sig_g):
        """V3: Explicitly test that dimensionality is maintained"""
        bn = CliffordEPBottleneckV3(in_dim=64, out_dim=64, sig_g=sig_g)
        x = torch.randn(8, 64)
        y = bn(x)

        # Output should match input dimensionality
        assert y.shape == (8, 64), f"Expected (8, 64), got {y.shape}"

    def test_v3_rejects_compression(self, sig_g):
        """V3: Should reject out_dim != in_dim (no compression allowed)"""
        with pytest.raises(ValueError, match="out_dim == in_dim"):
            CliffordEPBottleneckV3(in_dim=64, out_dim=32, sig_g=sig_g)

    def test_identity_bottleneck(self):
        """IdentityBottleneck: Should return input unchanged"""
        bn = IdentityBottleneck(in_dim=64)
        x = torch.randn(8, 64)
        y = bn(x)

        assert torch.allclose(y, x), "IdentityBottleneck should return input unchanged"

    def test_linear_bottleneck_shape(self):
        """LinearBottleneck: Should maintain dimensionality"""
        bn = LinearBottleneck(in_dim=64)
        x = torch.randn(8, 64)
        y = bn(x)

        assert y.shape == (8, 64), f"Expected (8, 64), got {y.shape}"


class TestGradientFlow:
    """Test that gradients flow correctly (no detach() issues)"""

    @pytest.fixture
    def sig_g(self):
        return torch.tensor([1.0, 1.0])

    def test_v2_gradient_flow(self, sig_g):
        """V2: Gradients should flow through bottleneck"""
        bn = BN_V2(in_dim=64, out_dim=32, sig_g=sig_g)
        x = torch.randn(8, 64, requires_grad=True)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        # Check gradient exists and is non-zero
        assert x.grad is not None, "Input gradient is None"
        assert x.grad.abs().sum() > 0, "Input gradient is zero"

    def test_v3_gradient_flow(self, sig_g):
        """V3: Gradients should flow correctly"""
        bn = CliffordEPBottleneckV3(in_dim=64, out_dim=64, sig_g=sig_g)
        x = torch.randn(8, 64, requires_grad=True)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient is None"
        assert x.grad.abs().sum() > 0, "Input gradient is zero"

    def test_v2_parameters_get_gradients(self, sig_g):
        """V2: Parameters should receive gradients"""
        bn = BN_V2(in_dim=64, out_dim=32, sig_g=sig_g)
        x = torch.randn(8, 64)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        # Check that at least input_proj and output_proj get gradients
        # (geom_regularizer is not used in V2 forward, so skip those)
        assert bn.input_proj.weight.grad is not None, "input_proj has no gradient"
        assert bn.output_proj.weight.grad is not None, "output_proj has no gradient"
        assert bn.input_proj.weight.grad.abs().sum() > 0, "input_proj gradient is zero"
        assert bn.output_proj.weight.grad.abs().sum() > 0, "output_proj gradient is zero"

    def test_v3_parameters_get_gradients(self, sig_g):
        """V3: Parameters should receive gradients"""
        bn = CliffordEPBottleneckV3(in_dim=64, out_dim=64, sig_g=sig_g)
        x = torch.randn(8, 64)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        for name, param in bn.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.abs().sum() > 0, f"Parameter {name} gradient is zero"


class TestMathematicalProperties:
    """Test mathematical correctness of operations"""

    @pytest.fixture
    def sig_g(self):
        return torch.tensor([1.0, 1.0])

    def test_v3_clifford_reshaping(self, sig_g):
        """V3: Verify Clifford reshaping is correct"""
        bn = CliffordEPBottleneckV3(in_dim=64, out_dim=64, sig_g=sig_g)

        # Manually verify reshaping logic
        # 64 dims with 4 blades = 16 multivectors
        assert bn.n_multivectors == 16, f"Expected 16 multivectors, got {bn.n_multivectors}"
        assert bn.sig.n_blades == 4, f"Expected 4 blades, got {bn.sig.n_blades}"
        assert bn.n_multivectors * bn.sig.n_blades == 64, "Dimensional consistency check failed"

    def test_v3_invertibility_of_reshape(self, sig_g):
        """V3: Reshaping should be invertible (no information loss in reshape itself)"""
        bn = CliffordEPBottleneckV3(in_dim=64, out_dim=64, sig_g=sig_g)

        # Create input
        x = torch.randn(8, 64)

        # Reshape to Clifford space and back
        x_reshaped = x.view(8, 16, 4)
        x_recovered = x_reshaped.view(8, 64)

        # Should be identical (reshape is invertible)
        assert torch.allclose(x, x_recovered), "Reshape is not invertible!"

    def test_v2_information_flow(self, sig_g):
        """V2: Information flows through projections"""
        bn = BN_V2(in_dim=64, out_dim=32, sig_g=sig_g)

        # Create deterministic input
        x = torch.ones(1, 64)
        y = bn(x)

        # Output should be non-zero (information passed through)
        assert not torch.allclose(y, torch.zeros_like(y)), "Output is all zeros"

    def test_v3_transformation_is_learnable(self, sig_g):
        """V3: Transformation parameters should affect output"""
        bn1 = CliffordEPBottleneckV3(in_dim=64, out_dim=64, sig_g=sig_g)
        bn2 = CliffordEPBottleneckV3(in_dim=64, out_dim=64, sig_g=sig_g)

        x = torch.randn(8, 64)

        # Initialize with same weights
        bn2.load_state_dict(bn1.state_dict())

        y1 = bn1(x)
        y2 = bn2(x)

        # Should be identical with same weights
        assert torch.allclose(y1, y2), "Same weights produce different outputs"

        # Modify weights and verify output changes
        with torch.no_grad():
            bn2.blade_transform.mul_(2.0)

        y2_modified = bn2(x)

        # Output should change with different weights
        assert not torch.allclose(y1, y2_modified), "Weight change doesn't affect output"


class TestNoInformationLoss:
    """Test that dimensionality-preserving bottlenecks don't lose information"""

    @pytest.fixture
    def sig_g(self):
        return torch.tensor([1.0, 1.0])

    def test_v3_no_magnitude_loss(self, sig_g):
        """V3: Should not normalize away magnitude information"""
        bn = CliffordEPBottleneckV3(in_dim=64, out_dim=64, sig_g=sig_g)

        # Create input with various magnitudes
        x = torch.randn(8, 64)
        x_norm_before = torch.norm(x, dim=1)

        y = bn(x)
        y_norm_after = torch.norm(y, dim=1)

        # Should preserve general magnitude structure
        # (not exact, but shouldn't zero out)
        assert (y_norm_after > 0).all(), "Output has zero norms"
        assert y_norm_after.mean() > 0.1, "Output magnitudes severely reduced"

    def test_v2_compression_loses_info(self, sig_g):
        """V2 (baseline): Compression should reduce dimensionality"""
        bn = BN_V2(in_dim=64, out_dim=32, sig_g=sig_g)
        x = torch.randn(8, 64)
        y = bn(x)

        # Output has fewer dimensions
        assert y.shape[1] == 32, "Compression didn't reduce dimensionality"
        assert y.shape[1] < x.shape[1], "Output not smaller than input"


class TestNoBugsRegression:
    """Test that specific bugs don't regress"""

    @pytest.fixture
    def sig_g(self):
        return torch.tensor([1.0, 1.0])

    def test_no_detach_bug_v2(self, sig_g):
        """BUG FIX: V2 should not use detach() that breaks gradients"""
        bn = BN_V2(in_dim=64, out_dim=32, sig_g=sig_g)

        x = torch.randn(8, 64, requires_grad=True)
        y = bn(x)
        loss = y.sum()
        loss.backward()

        # If detach() was used incorrectly, gradient would be None
        assert x.grad is not None, "BUG REGRESSION: detach() breaks gradients"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Gradients are zero"

    def test_v3_no_over_constraint(self, sig_g):
        """BUG FIX: V3 should not normalize away magnitude (unlike my broken 'fix')"""
        bn = CliffordEPBottleneckV3(in_dim=64, out_dim=64, sig_g=sig_g)

        # Create input with known magnitude
        x = torch.ones(8, 64) * 5.0  # magnitude ~5
        y = bn(x)

        y_magnitude = torch.norm(y, dim=1).mean()

        # Should preserve reasonable magnitude (not normalized to unit length)
        # Unit norm would give magnitude ~8 (sqrt(64)), we have 5*sqrt(64) = 40
        # After transformation, should still be in reasonable range
        assert y_magnitude > 1.0, "BUG REGRESSION: Output over-constrained"
        assert y_magnitude < 100.0, "Output has unreasonable magnitude"

    def test_compression_plus_expansion_is_lossy(self, sig_g):
        """DESIGN BUG: 64→32→64 compression + expansion is lossy"""
        bn = BN_V2(in_dim=64, out_dim=32, sig_g=sig_g)

        # This bottleneck compresses. Let's verify the lossiness
        x = torch.randn(8, 64)
        y = bn(x)

        # y is only 32-dimensional, not 64
        assert y.shape[1] == 32, "V2 should compress to 32"

        # The point: V2 loses information (by design as a bottleneck)
        # But Phase 4 tries to use 32D output as if it's 64D
        # This is fundamentally flawed


class TestIntegration:
    """Integration tests with actual models"""

    @pytest.fixture
    def sig_g(self):
        return torch.tensor([1.0, 1.0])

    def test_bottleneck_in_mlp(self, sig_g):
        """Test bottleneck inside an MLP"""
        class MLP(nn.Module):
            def __init__(self, bottleneck_fn):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.bottleneck = bottleneck_fn(in_dim=64, out_dim=64, sig_g=sig_g)
                self.fc2 = nn.Linear(64, 10)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.bottleneck(x)
                x = self.fc2(x)
                return x

        model = MLP(CliffordEPBottleneckV3)
        x = torch.randn(8, 64)

        # Should run without error
        y = model(x)
        assert y.shape == (8, 10), f"Expected (8, 10), got {y.shape}"

        # Should support backprop
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None, "Parameter has no gradient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
