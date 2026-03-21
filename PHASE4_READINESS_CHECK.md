# Phase 4 Readiness Checklist

**Date:** 2026-03-20  
**Status:** ✅ READY TO PROCEED

---

## Pre-Phase 4 Validation

### ✅ Critical Path Validation
- [x] Geometric algebra operations corrected (geometric_product bug fixed)
- [x] Phase 2 structural explorations (5/6 working)
- [x] Phase 3 domain benchmarks (4/4 working, results validated)
- [x] P2.9 Bottleneck layer (production-ready, 16.4% improvement CartPole, 79.4% MSE N-body)
- [x] Integration tests (P2.9 on Phase 3 domains working)

### ✅ Code Quality
- [x] All experiments re-validated with corrected algebra
- [x] No inflated results (accounted for bug fixes)
- [x] Known limitations documented (P2.10 architectural)
- [x] Debug statements removed
- [x] Clean git history with clear commit messages

### ✅ Infrastructure Ready
- [x] P2.9 Bottleneck model: `cliffeq/models/bottleneck_v2.py` ✓
- [x] Energy functions: GraphEnergy, BilinearEnergy, etc. ✓
- [x] Dynamics rules: LinearDot, GeomProduct, etc. ✓
- [x] Training engines: EP, CHL, FF, CD ✓
- [x] Metrics harness: equivariance, convergence tracking ✓

---

## Phase 4 Objectives

### Primary Goal
**Prove P2.9 is a universal geometric processing primitive by inserting it into baseline architectures across domains without other modifications.**

### Test Matrix

| Domain | Baseline | Task | Metric |
|--------|----------|------|--------|
| **Vision** | ResNet-18 | CIFAR-10 + rotation | Rotated accuracy, equivariance |
| **Language** | Transformer-2L | SST-2 sentiment | Accuracy, OOD robustness |
| **RL** | PPO MLP | CartPole mirror | Mirror-invariant reward |
| **Graphs** | GCN-2L | MUTAG classification | Accuracy |

### Success Criteria
- [ ] P2.9 improves ≥2 of 4 domains without baseline modification
- [ ] Improvement >5% (not just noise)
- [ ] Parameter-matched ablations confirm Clifford+EP advantage
- [ ] OOD generalization validated (rotation shift, domain shift)

---

## Ready-To-Use Components

### Models
- ✓ `CliffordEPBottleneckV2` — Drop-in layer for any architecture
- ✓ Baseline implementations (ResNet, Transformer, PPO, GCN)

### Energy Functions
- ✓ `SimpleGeometricEnergy` — Self-energy (used in P2.9)
- ✓ `BilinearEnergy` — Bilinear form
- ✓ `GraphEnergy` — Graph-structured

### Training
- ✓ `EPEngine` — Equilibrium Propagation training
- ✓ EP training integrated with supervised loss (via clamped phase)

### Metrics
- ✓ `equivariance_violation(model, x, group)` — SO(2), SO(3), O(3), Z2
- ✓ `convergence_curve(energy, dynamics, x_init, n_steps)`
- ✓ `fixed_point_count(energy, dynamics, n_init=200)`
- ✓ Wandb integration for experiment tracking

---

## Implementation Plan

### Phase 4.1: Vision (2-3 days)
1. Implement ResNet-18 + P2.9 bottleneck
2. CIFAR-10 baseline training
3. Rotation robustness testing (0°, 90°, 180°, 270°)
4. Equivariance metrics

### Phase 4.2: Language (2-3 days)
1. Implement Transformer-2L + P2.9 bottleneck
2. SST-2 baseline training
3. OOD domain shift testing (different review sources)
4. Sentiment-invariant embedding analysis

### Phase 4.3: RL (2-3 days)
1. PPO baseline for CartPole
2. P2.9 bottleneck integration (after hidden layer)
3. Mirror symmetry test (policy should generalize)
4. Mirror-invariant reward plotting

### Phase 4.4: Graphs (1-2 days)
1. GCN-2L + P2.9 bottleneck
2. MUTAG classification
3. Permutation invariance check

### Phase 4.5: Analysis & Reporting (2 days)
1. Aggregate results across all domains
2. Parameter-matched ablation analysis
3. OOD generalization summary
4. Publication-ready figures and tables

---

## Confidence Assessment

| Aspect | Confidence | Notes |
|--------|-----------|-------|
| Geometric algebra correctness | **HIGH** | Validated in Phases 2-3 |
| P2.9 bottleneck quality | **HIGH** | 16-79% improvements in Phase 3 |
| Infrastructure completeness | **HIGH** | All training engines implemented |
| Cross-domain applicability | **MEDIUM** | Hypothesis; Phase 4 tests it |
| Publication readiness | **MEDIUM** | Depends on Phase 4 results |

---

## Risk Mitigation

### Risks
- P2.9 might not generalize across domains
- OOD evaluation might be too noisy
- Parameter matching could be tricky

### Mitigation
- Use conservative evaluation (multiple seeds, significance tests)
- Ensemble multiple evaluation metrics
- Document any negative results thoroughly
- Ablations clear up whether improvement is from Clifford or EP

---

## Success Indicators

### Tier 1: Publication-Ready
- ✓ P2.9 improves ≥3 domains by ≥5%
- ✓ OOD robustness validated
- ✓ Ablations show Clifford+EP advantage
- ✓ Clean narrative: "Geometric processing as universal bottleneck"

### Tier 2: Promising Results
- ✓ P2.9 improves ≥2 domains by ≥5%
- ✓ At least one domain shows strong OOD robustness
- ✓ Narrative: "Limited but meaningful improvements"

### Tier 3: Technical Contribution
- ✓ Framework is solid and extensible
- ✓ All code is documented and reproducible
- ✓ Negative results are interpretable

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1-2: Toy models | ✓ DONE | Foundation |
| Phase 3: Domain baselines | ✓ DONE | Validation |
| **Phase 4: Bottleneck test** | **5-10 days** | READY TO START |
| Final analysis + publication | **2-3 days** | After Phase 4 |

**Total to publication:** ~10-15 days from now

---

## Decision

### ✅ READY TO PROCEED WITH PHASE 4

All prerequisites met:
- Core technology validated (Clifford algebra + P2.9)
- Supporting infrastructure complete
- Hypothesis clear and testable
- Success criteria defined
- Timeline realistic

**Recommendation:** Proceed immediately to Phase 4.1 (Vision domain).

---

**Approved:** 2026-03-20 20:45 UTC  
**Next Step:** Implement ResNet-18 + P2.9 bottleneck for CIFAR-10 rotation testing
