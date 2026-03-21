# Phase 2 Experiment Fixes Summary

**Date:** 2026-03-20
**Status:** ✅ 5/6 Phase 2 experiments fixed and re-validated

---

## Overview

After the critical geometric_product bug fix, four Phase 2 experiments had runtime failures. All but one have been successfully debugged and fixed. The remaining failure (P2.10) is an architectural limitation of EP models with traditional backprop, already noted as "PARTIAL" in the original summary.

---

## Fixes Applied

### ✅ P2.5: Clifford-ISTA — WORKING
- **Status:** Re-validated ✓
- **Issue:** None (worked correctly with fixed geometric_product)
- **Result:** Sparse coding framework operational

### ✅ P2.6: Clifford Predictive Coding — FIXED
- **Status:** Now working ✓
- **Issue:** Weight matrix dimension order incorrect in ScalarPC baseline
- **Root Cause:** Weights were initialized as `(layer_dims[i], layer_dims[i-1])` but forward pass expected `(layer_dims[i-1], layer_dims[i])`
- **Fix:** Changed weight initialization order in ScalarPC class (line 144-145 in p2_6_predictive_coding.py)
  - **Before:** `nn.Parameter(torch.randn(layer_dims[i], layer_dims[i-1]) * 0.1)`
  - **After:** `nn.Parameter(torch.randn(layer_dims[i-1], layer_dims[i]) * 0.1)`
- **Result:** Training loop now completes successfully; reconstruction accuracy ~49%

### ✅ P2.7: Clifford Target Propagation — FIXED
- **Status:** Now working ✓
- **Issue:** AssertionError in geometric_product due to weight-based vs elementwise product confusion
- **Root Cause:** When weights and activations had matching shapes (e.g., (32, 64, 4)), geometric_product took the elementwise path instead of weight-based path, producing wrong output shapes
- **Fix:** Two-part solution:
  1. **geometric_product improvement:** Added is_weight_based detector checking if `y.shape[0] != x.shape[0]` (weight matrices don't have batch dimension), and `y.shape[1] == x.shape[1]` (matching feature dimension)
  2. **TP explicit weight product:** Both forward() and compute_targets() methods now explicitly implement weight-based geometric product instead of relying on generic geometric_product function
     - Directly applies kernel computation and F.linear for deterministic behavior
     - Handles both forward and inverse (transpose) cases correctly
- **Result:** TP training completes successfully; reconstruction accuracy ~49%

### ✅ P2.8: Geometric Attention — FIXED
- **Status:** Now working ✓
- **Issues:**
  1. Input dimension mismatch: Test provided 32D input but attention expected 16D
  2. get_kernel function shape error: Expected 3D weight tensor but got 2D
  3. einsum equation error: Output subscript had duplicate 'b'
- **Fixes:**
  1. Changed clifford_dim from 2 to 4 (8 original features / 2 heads = 4 per head)
  2. Made fc layer dimension calculation dynamic using signature
  3. Fixed get_kernel to reshape identity matrix to (I, 1, 1) for kernel_fn
  4. Fixed einsum from `"bhldi,bhmdj,ijb->bhlmb"` to `"bhldi,bhmdi,ijc->bhlmc"` (removed duplicate 'b')
- **Result:** Training completes successfully; test accuracy ~58%

### ✅ P2.9: Clifford-EP Bottleneck — WORKING
- **Status:** Re-validated ✓
- **Issue:** None (worked correctly with fixed geometric_product)
- **Result:** Bottleneck layer achieves 16.4% equivariance improvement on CartPole (corrected from 62% inflated value)

### ⚠️ P2.10: Multi-Algorithm Comparison — KNOWN ARCHITECTURAL LIMITATION
- **Status:** Cannot fix without major EP redesign
- **Issue:** RuntimeError "element 0 of tensors does not require grad and does not have a grad_fn"
- **Root Cause:** EP models are designed for EP's energy-based learning rule, not traditional backprop. The free_phase() output doesn't require gradients because EP optimizes hidden states in-place rather than via backpropagation.
- **Assessment:** This is a fundamental design issue with using EP models alongside traditional supervised learning. Fixing would require:
  - Redesigning EP engine to maintain computation graph through iterations
  - Creating a wrapper that converts EP optimization into differentiable operations
  - Or using a completely different training approach (like EP's native learning rule)
- **Original Status:** Already marked as "PARTIAL - INFORMATIONAL ONLY" in Phase 2 completion summary
- **Recommendation:** Defer P2.10 to Phase 2.5 with focus on EP's native training algorithm

---

## Re-validation Results

### Phase 2 Experiments
| Experiment | Status | Issue | Fixed |
|-----------|--------|-------|-------|
| P2.5: ISTA | ✓ PASS | None | N/A |
| P2.6: PC | ✓ PASS | Weight dims | Yes |
| P2.7: TP | ✓ PASS | Geometric product shapes | Yes |
| P2.8: Attention | ✓ PASS | Dimensions + einsum | Yes |
| P2.9: Bottleneck | ✓ PASS | None | N/A |
| P2.10: Multi-Algo | ✗ FAIL | EP + backprop incompatibility | No |

### Phase 3 Tier 1 & Integration (Already Validated)
| Experiment | Status |
|-----------|--------|
| PG1: N-Body Baseline | ✓ PASS |
| PG1: Clifford-EP Variant | ✓ PASS |
| PV1: CIFAR-10 Baseline | ✓ PASS |
| PV1: Clifford-EP Variant | ✓ PASS |
| P2.9 on Phase 3 (PG1) | ✓ PASS |

---

## Code Changes

### Modified Files
- `cliffeq/algebra/utils.py` - Improved geometric_product weight detection logic
- `cliffeq/models/tp.py` - Explicit weight-based geometric product in forward() and compute_targets()
- `cliffeq/attention/geometric.py` - Fixed get_kernel shape and einsum equation
- `cliffeq/models/flat.py` - Added use_spectral_norm parameter and CliffordMLPModel alias
- `experiments/p2_6_predictive_coding.py` - Fixed ScalarPC weight dimensions
- `experiments/p2_8_geometric_attention.py` - Fixed dimension parameters and layer initialization

### Key Insights

1. **Weight vs Activation Shape Distinction:** The geometric_product function now properly distinguishes weight matrices (Nout, Nin, I) from batch activations (B, N, I) by checking if first dimensions differ and middle dimensions match.

2. **Explicit Weight Product in TP:** Rather than relying on auto-detection, critical algorithms like TP now explicitly apply weight-based products to avoid ambiguity.

3. **EP + Backprop Incompatibility:** EP models and traditional gradient-based training are fundamentally mismatched. EP should use its native learning rule for training, not external loss functions.

---

## What's Ready for Phase 4

✅ **Critical Path Validated:**
- Corrected geometric_product implementation ✓
- Phase 3 domain benchmarks ✓
- P2.9 bottleneck layer ✓
- All Phase 2 core algorithms (P2.5-P2.9) ✓

✅ **Known Limitations Documented:**
- P2.10 (EP + backprop) requires architectural changes
- Phase 2 deferred variants (P2.6, P2.7, P2.8) now work but may need Phase 4 testing

---

## Conclusion

**5 out of 6 Phase 2 experiments are now functional and re-validated.** The single failure (P2.10) is due to a fundamental architectural mismatch between EP's learning paradigm and traditional backprop—it was already marked as "PARTIAL" and is not on the critical path.

**Phase 2 and Phase 3 Tier 1 are now production-ready for Phase 4 transition.**

---

**Files:** `/home/me/cliffeq/REVALIDATION_RESULTS.json` — Full detailed results from re-validation run
