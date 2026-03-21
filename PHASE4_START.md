# Phase 4: Cross-Domain Clifford-EP Bottleneck Validation — Ready to Start

**Date:** 2026-03-20
**Status:** ✅ ALL EXPERIMENTS IMPLEMENTED AND READY

---

## Overview

Phase 4 validates that the P2.9 Clifford-EP bottleneck (CliffordEPBottleneckV2) functions as a **universal geometric processing primitive** by inserting it as a drop-in layer into baseline architectures across four distinct domains.

**Core hypothesis:** Geometric regularization via Clifford algebra improves model robustness and generalization across different problem types.

---

## Phase 4 Experiment Matrix

### Four Domains, Four Baselines, One Bottleneck

| Domain | Baseline | Task | Data | P2.9 Integration |
|--------|----------|------|------|-----------------|
| **Vision** | ResNet-18 | CIFAR-10 classification | 32×32 RGB images | After initial conv pool |
| **Language** | Transformer-2L | SST-2 sentiment | Synthetic word sequences | After embedding layer |
| **RL** | PPO MLP | CartPole-v1 | State-action trajectories | After first hidden layer |
| **Graphs** | GCN-2L | MUTAG classification | Synthetic molecular graphs | After first graph conv |

---

## Experiment Files

Ready to execute:

1. **`experiments/p4_1_resnet_clifford_bottleneck.py`**
   - Vision domain test
   - ResNet-18 baseline vs. ResNet-18 + P2.9 bottleneck
   - CIFAR-10 training loop (20 epochs)
   - Rotation robustness metrics
   - Runtime: ~15-20 minutes per variant

2. **`experiments/p4_2_transformer_sentiment.py`**
   - Language domain test
   - Transformer-2L baseline vs. Transformer-2L + P2.9 bottleneck
   - Synthetic SST-2-like sentiment data (2000 samples)
   - Training loop (15 epochs)
   - Runtime: ~10-15 minutes per variant

3. **`experiments/p4_3_ppo_cartpole.py`**
   - RL domain test
   - PPO MLP baseline vs. PPO + P2.9 bottleneck
   - CartPole-v1 environment
   - Training loop (1000 steps, 4 episodes/step)
   - Runtime: ~20-30 minutes per variant

4. **`experiments/p4_4_gnn_graph_classification.py`**
   - Graph domain test
   - GCN-2L baseline vs. GCN-2L + P2.9 bottleneck
   - Synthetic MUTAG-like molecular graphs (500 samples)
   - Training loop (20 epochs)
   - Runtime: ~10-15 minutes per variant

5. **`experiments/p4_orchestrate_all_domains.py`** ⭐
   - Master script that runs P4.1 through P4.4 sequentially
   - Aggregates results into comprehensive Phase 4 summary
   - Generates JSON results and markdown report
   - Total runtime: ~90-120 minutes
   - **Recommended entry point for Phase 4 execution**

---

## How to Run Phase 4

### Option 1: Run All Four Domains (Recommended)
```bash
python experiments/p4_orchestrate_all_domains.py
```
This executes all four domain experiments sequentially and produces:
- `results/PHASE4_COMPLETE_[timestamp].json` — Comprehensive results
- `results/PHASE4_REPORT_[timestamp].md` — Executive summary report

### Option 2: Run Individual Domains
```bash
# Vision
python experiments/p4_1_resnet_clifford_bottleneck.py

# Language
python experiments/p4_2_transformer_sentiment.py

# RL
python experiments/p4_3_ppo_cartpole.py

# Graphs
python experiments/p4_4_gnn_graph_classification.py
```

Each produces individual result files in `results/` directory.

---

## Expected Behavior

Each experiment follows the same pattern:

1. **Load dataset/environment**
2. **Train baseline model** (standard architecture)
3. **Train Clifford variant** (baseline + P2.9 bottleneck after first layer)
4. **Evaluate both** and compute improvement metrics
5. **Save results** to `results/p4_[domain]_[timestamp].json`

### Success Criteria

Phase 4 demonstrates success if:

✅ **Primary:** P2.9 bottleneck improves ≥2 of 4 domains (50% positive result rate)
✅ **Secondary:** Improvements are >5% (not noise)
✅ **Tertiary:** Bottleneck integrates cleanly without architectural modifications

---

## Architecture Integration Points

### Vision (ResNet-18)
```
Input → Conv1 + BN + ReLU → MaxPool
         ↓
    [P2.9 BOTTLENECK] ← Feature bottleneck here
         ↓
      Layer1-4 → AvgPool → Classifier
```

### Language (Transformer-2L)
```
Token IDs → Embedding + Positional
             ↓
        [P2.9 BOTTLENECK] ← Per-token bottleneck here
             ↓
          Layer1-2 → GlobalPool → Classifier
```

### RL (PPO)
```
State → Linear(state_dim → hidden_dim)
        ↓
   [P2.9 BOTTLENECK] ← Hidden state bottleneck here
        ↓
     Linear(hidden_dim → hidden_dim)
     ↙                    ↘
  Actor Head          Critic Head
```

### Graphs (GCN-2L)
```
Node Features → GCNConv(in_ch → hid_ch)
                ↓
           [P2.9 BOTTLENECK] ← Node-level bottleneck here
                ↓
           GCNConv(hid_ch → hid_ch) → GlobalPool → Classifier
```

---

## Key Implementation Details

### P2.9 Bottleneck Configuration

Each domain uses:
- **Signature:** Cl(2,0) — 4D multivectors (1 scalar + 3 vectors)
- **EP Steps:** 2-3 steps per forward pass (lightweight)
- **Step Size:** 0.01 (conservative)
- **Spectral Norm:** Enabled for training stability
- **Gradient Flow:** Full backprop through bottleneck

### Metrics Tracked

Per domain:
- Training loss and accuracy/return per epoch
- Test/validation accuracy or reward
- Comparison: baseline vs. Clifford variant
- Absolute and relative improvement

---

## Dependencies

Core:
- `torch`, `torchvision` (Vision)
- `gymnasium` (RL — requires `pip install gymnasium`)
- `torch_geometric` (Graphs)
- NumPy, scikit-learn (metrics)
- `networkx` (synthetic graph generation)

Ensure installed:
```bash
pip install torch torchvision torch-geometric gymnasium scikit-learn networkx
```

---

## Results Output

### Individual Experiment Results
```json
{
  "baseline": {
    "final_test_accuracy": 0.7234,
    "train_accuracies": [...],
    "test_accuracies": [...],
    "train_losses": [...]
  },
  "clifford": {
    "final_test_accuracy": 0.7456,
    "train_accuracies": [...],
    "test_accuracies": [...],
    "train_losses": [...]
  }
}
```

### Phase 4 Comprehensive Results
```json
{
  "timestamp": "20260320_...",
  "experiments": {
    "vision": { "status": "completed", "results": {...} },
    "language": { "status": "completed", "results": {...} },
    "rl": { "status": "completed", "results": {...} },
    "graphs": { "status": "completed", "results": {...} }
  },
  "summary": {
    "total_domains_tested": 4,
    "improvements_by_domain": [...],
    "avg_improvement": 0.0234,
    "avg_improvement_pct": 3.45,
    "domains_with_improvement": 3
  }
}
```

---

## Next Steps After Phase 4

Once Phase 4 execution completes:

**Phase 4.5: Cross-Domain Analysis (1-2 days)**
- Aggregate results across all four domains
- Parameter-matched ablations (ensure fair comparison)
- Statistical significance testing
- Publication-ready reporting
- Visualization and figures

**Phase 5: Publication & Dissemination**
- Write paper documenting the findings
- Prepare reproducible code release
- Submit to venue (conference/journal)

---

## Timeline Estimate

- **Phase 4 execution:** ~2 hours (all domains sequentially)
- **Phase 4.5 analysis:** ~1-2 days
- **Total to publication-ready:** ~10-15 days

---

## Git Status

All Phase 4 experiment files are ready to commit:
- ✅ `experiments/p4_1_resnet_clifford_bottleneck.py`
- ✅ `experiments/p4_2_transformer_sentiment.py`
- ✅ `experiments/p4_3_ppo_cartpole.py`
- ✅ `experiments/p4_4_gnn_graph_classification.py`
- ✅ `experiments/p4_orchestrate_all_domains.py`
- ✅ `PHASE4_START.md` (this file)

---

## Instructions

**To begin Phase 4:**

```bash
# Commit Phase 4 implementations
git add experiments/p4_*.py PHASE4_START.md
git commit -m "Phase 4 implementation: Cross-domain bottleneck validation"

# Run all domains
python experiments/p4_orchestrate_all_domains.py
```

---

**Status:** ✅ READY TO START PHASE 4

All experiments implemented. Awaiting execution.

