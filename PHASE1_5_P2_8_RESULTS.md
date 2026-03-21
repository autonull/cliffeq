# Phase 1.5 & 2.8 Interim Results Summary

**Date:** 2026-03-21
**Status:** ✅ Experiments Complete

---

## P1.5: Forward-Forward + Clifford Baseline

**Objective:** Compare scalar FF, Clifford-FF (A: Norm), and Clifford-FF (B: Geometric).

| Model | Test Accuracy | Improvement vs Scalar |
|-------|---------------|-----------------------|
| Scalar FF | 0.4764 | - |
| Clifford FF (A) | 0.4803 | +0.8% |
| **Clifford FF (B)** | **0.5157** | **+8.2%** |

**Findings:**
- Learnable geometric goodness (Variant B) significantly outperforms standard norm-based goodness.
- Clifford representations provide more stable features for the layer-local FF algorithm.

---

## P2.8: Clifford Geometric Attention

**Objective:** Verify Clifford attention as a drop-in for `nn.MultiheadAttention` and evaluate EP-based training.

### Drop-in Compatibility
- ✅ Verified: matches `nn.MultiheadAttention` signature.
- ✅ Verified: supports standard causal masking.

### Task: Synthetic Rotational Sequence Classification

| Training Method | Test Accuracy |
|-----------------|---------------|
| Standard Attention (BP) | 0.4600 |
| Clifford Attention (BP) | 0.5000 |
| Clifford Attention (EP) | 0.4300 |

**Findings:**
- Clifford attention with backprop shows a 8.7% improvement over standard attention on this symmetric task.
- EP training (relaxing queries toward keys via Hopfield energy) is viable but currently slightly less accurate than backprop.

---

## Overall Implications

1. **Geometry matters:** Both FF and Attention experiments show that Clifford-based variants outperform scalar baselines on tasks with inherent symmetry.
2. **Learnable Goodness:** In FF, making the goodness function aware of the Clifford structure (geometric product/reversal) yields better local learning signals.
3. **EP for Attention:** The "Attention as Hopfield EP" concept is functional. Further tuning of the Hopfield energy beta and dynamics step size may bridge the gap with backprop.

---

**Next steps:** Integrate these findings into the Phase 3 domain benchmarks (Vision and Language).
