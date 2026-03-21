# Phase 3 Initial Results: PG1 & PV1 Baselines

**Date:** 2026-03-20
**Status:** PG1 and PV1 baselines established; ready for Clifford-EP integration

---

## Executive Summary

Phase 3 Tier 1 critical path experiments (PG1: N-body dynamics, PV1: CIFAR-10 rotation) have been implemented and executed with standard baselines. These results establish the performance baseline against which Clifford-EP variants will be compared.

**Key Findings:**
- **PG1 (N-body):** Standard MLP baseline achieves excellent accuracy (MSE: 0.000007) with low equivariance violation (0.001125)
- **PV1 (CIFAR-10):** Standard CNN achieves 47.8% accuracy with 47.32% equivariance violation; rotation-invariant approach trades accuracy for equivariance
- **Framework Status:** Both domains now have working data pipelines, baselines, and equivariance metrics ready for Clifford-EP variants

---

## PG1: N-Body Dynamics Benchmark

### Task Description
Predict next particle positions in Coulomb force N-body systems (5 particles).
- Training: 400 samples, Test: 100 samples
- Equivariance test: SO(3) rotations on particle positions

### Results

| Model | Test MSE | Equivariance Violation |
|-------|----------|----------------------|
| **Baseline (hidden=64)** | 0.000007 | 0.001125 |
| Baseline (hidden=128) | 0.000014 | 0.001362 |

### Key Observations

1. **Baseline performance is strong:** Simple MLP converges quickly (loss drops from 0.0005 to 0.00002 in 4 epochs)

2. **Equivariance is naturally good:** Both models maintain <0.15% disagreement under rotations, suggesting the task has inherent structure

3. **Higher capacity hurts equivariance:** Adding more hidden units (64→128) degrades rotation robustness by 21%, indicating overfitting to spurious features

4. **Implication for Phase 3:** Clifford-EP should be tested on this domain to see if geometric representation can maintain accuracy while further improving equivariance

### Next Steps

- Implement Clifford-EP version using BilinearEnergy (requires proper gradient flow through EP iterations)
- Compare: Clifford-EP vs. standard MLP on accuracy and equivariance metrics
- Test on larger systems (20 particles) if phase 3 time permits

---

## PV1: CIFAR-10 Rotation Invariance Benchmark

### Task Description
CIFAR-10 classification with evaluation on rotated test images (15°, 30°, 45°, 60°, 90°).
- Training: 5000 samples, Test: ~1000 samples
- Equivariance test: Random rotation at test time
- Metric: Rotation robustness (avg accuracy on rotated images)

### Results

| Model | Accuracy | Rotation Robustness | Equivariance Violation |
|-------|----------|---------------------|----------------------|
| **Baseline CNN** | 0.4780 | 0.3306 | 0.4732 |
| Variant CNN (larger) | 0.4700 | 0.3198 | 0.4934 |
| Rotation-Invariant (4x rotation averaging) | 0.3440 | 0.3358 | 0.1324 |

### Key Observations

1. **Standard CNNs are NOT rotation invariant:** Accuracy drops 30% (47.8% → 33%) on rotated inputs

2. **Rotation averaging works for equivariance:** Using 4 rotations (0°, 90°, 180°, 270°) reduces equivariance violation from 47% to 13%, but at cost of 13.8% accuracy drop

3. **Capacity doesn't help equivariance:** Larger variant (64→128 conv filters) actually hurts both accuracy and equivariance vs. standard baseline

4. **Implication for Phase 3:** Clifford-EP with geometric representation may naturally provide rotation awareness without 4x computational cost of averaging. Test on this domain is critical.

### Next Steps

- Implement Clifford-EP CNN using multivector representations for spatial features
- Compare: Standard CNN vs. Clifford-EP vs. Rotation-Invariant on accuracy/equivariance trade-off
- Evaluate if Clifford geometric product provides rotation structure naturally

---

## P2.9 Bottleneck Status

**Current Issue:** EP integration into standard architectures requires proper gradient flow through free phase iterations, which conflicts with PyTorch autograd.

**Approach for Phase 3:**
- Defer complex bottleneck integration (5+ hours debugging)
- Focus on domain-specific Clifford-EP models (PG1, PV1) that directly use EP training
- Return to P2.9 bottleneck once domain results are in hand (may inform the design)

---

## Technical Infrastructure Ready

✅ **Data Pipelines:**
- N-body: Coulomb force synthesis + SO(3) equivariance testing
- Vision: CIFAR-10 loading + rotation transforms + equivariance metrics

✅ **Baseline Models:**
- Simple MLP, CNNs with standard training

✅ **Evaluation Metrics:**
- MSE/accuracy
- Equivariance violation (misalignment under symmetric transforms)
- Rotation robustness (performance on rotated inputs)

✅ **Experiment Harnesses:**
- Training loops with proper device handling
- Evaluation on original + rotated/transformed data
- Comparative reporting

---

## Recommendations for Next Phase

### Immediate (This Week)

1. **PG1 Clifford-EP:** Implement domain-specific Clifford-EP for N-body
   - Use GeometricEquilibriumGNN or direct EPModel with proper gradient handling
   - Goal: Match or exceed baseline MSE while improving equivariance

2. **PV1 Clifford-EP:** Implement Clifford CNN variant
   - Use multivector spatial features in convolutional layers
   - Compare on accuracy/equivariance trade-off

3. **P2.9 Decision:** After seeing PG1/PV1 Clifford results, decide whether bottleneck is worth pursuing or if domain-specific approaches are sufficient

### Decision Point (After Domain Results)

- If Clifford-EP improves ≥2 domains → Investigate P2.9 bottleneck integration (high ROI)
- If Clifford-EP helps in only 1 domain → Focus on understanding why; may need hypothesis refinement
- If Clifford-EP shows no advantage → Reconsider framework before Phase 4

---

## Code Status

**New Experiments:**
- `experiments/p3_1_nbody_dynamics.py` (387 lines): N-body baseline with equivariance testing
- `experiments/p3_2_cifar10_rotation.py` (352 lines): CIFAR-10 with rotation robustness testing

**Ready for Integration:**
- Clifford-EP models from Phase 2 (sparse coding, GNN, etc.)
- BilinearEnergy and other energy functions from Phase 2 zoo

**Technical Debt:**
- P2.9 bottleneck requires careful integration of EP with standard architectures
- Future work: Consider layer-wise EP vs. end-to-end training for better gradient flow

---

## Timeline for Phase 3

**Day 1 (Today):** ✅ Establish baselines (PG1, PV1)
**Day 2-3:** Implement Clifford-EP variants for PG1 and PV1
**Day 4:** Evaluate results; make P2.9 go/no-go decision
**Day 5+:** Optional - scale to Phase 3 Tier 2 (PG2, PV2, PL1)

---

**Status:** Phase 3 foundation established. Ready to integrate Clifford-EP for domain-specific testing.
