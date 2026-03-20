# Clifford-EP Research Framework: TODO

---

## 1. Central Vision

The goal is to discover **general-purpose novel ML approaches** that can be applied across vision, language, reinforcement learning, and scientific domains — by combining Clifford geometric algebra with non-backpropagation training methods. The ideal outcome is not a specialized geometric model for robotics; it is a new class of building blocks that improve any architecture by giving it geometric processing, energy-based learning, or both.

The hypothesis is that geometry and energy minimization are underexploited inductive biases that can help across all domains — not just the ones where geometry is obvious. Text has syntactic and compositional structure that may benefit from geometric products. Images have symmetry structure that current CNNs handle clumsily. RL agents operate in worlds with reflection and rotation symmetries. The question is how far the "geometric energy minimization" idea can be pushed before the inductive bias becomes a liability.

---

## 2. Central Hypothesis

> **The Clifford Advantage Hypothesis:** For tasks with any form of hidden geometric, relational, or compositional structure — including tasks not obviously geometric — replacing scalar states with Clifford multivectors and replacing backpropagation with energy-based training will improve at least one of the following: sample efficiency, out-of-distribution generalization, symmetry equivariance, or training stability. The combination is more powerful than either ingredient alone.

This is the testable claim that drives every experiment. Each PoC should be designed to either confirm, refine, or falsify some aspect of it.

---

## 3. The Complete Design Space

The novel research territory is defined by four orthogonal axes. Explored cells are marked ✓.

### Axis A: State Representation

| Representation | Notes |
|---|---|
| Scalar (reals) ✓ | Standard; all existing DL |
| Grade-1 only (vectors) ✓ | Standard embeddings |
| Grade-2 truncated Cl(3,0), 7D | 1+3+3; primary target |
| Full Cl(3,0), 8D | Includes pseudoscalar |
| Rotor-only (even subalgebra) ≅ H₁ | Pure SO(3); states are rotations |
| Cl(1,3) Minkowski | Time + space; causal structure |
| PGA Cl(3,0,1) | Projective: translations + rotations as motors |
| Cl(p,q) adaptive | Grade and signature learned per layer |

### Axis B: Training Algorithm

| Algorithm | Backprop-free? | Notes |
|---|---|---|
| Backpropagation ✓ | No | Reference baseline |
| Equilibrium Propagation (EP) ✓ | Yes | Scellier & Bengio 2017; free + clamped phases |
| Contrastive Hebbian Learning (CHL) | Yes | EP variant; positive + negative phase |
| **Forward-Forward (FF)** | Yes | Hinton 2022; layer-local goodness maximization |
| Predictive Coding (PC) | Yes | Friston; minimize prediction error layer-locally |
| Target Propagation (TP) | Partial | Compute layer targets; no global gradient |
| ISTA / Sparse Coding | Yes | Energy minimization via soft thresholding |
| Contrastive Divergence (CD) | Yes | EBM training; MCMC-based negative phase |
| Modern Hopfield / Dense Associative Memory | Yes | Energy-based associative retrieval |
| Node/Weight Perturbation | Yes | Finite-difference gradient estimate |

### Axis C: Update Dynamics (for iterative methods)

| Rule | Formula | Cost |
|---|---|---|
| LinearDot | x ← x − α (∇E · x) | O(n) |
| GeomProduct | x ← x − α (∇E ✶ x) | O(n²) |
| ExpMap | x ← exp(−α ∇E) ✶ x | O(n³) |
| RotorOnly | Update even subalgebra as quaternion; rest linear | O(n) |
| Riemannian | Project ∇E onto tangent space of ∥x∥=1 sphere | O(n) |
| GradeSplit | Different step size α_k per grade | O(n) + tuning |
| WedgeUpdate | x ← x − α (∇E ∧ x) | O(n²) |

### Axis D: Application Domain

Vision · Language · Reinforcement Learning · Physics/Geometry · Graphs · Scientific simulation

**The novel research is in non-trivial intersections of all four axes.** Plain Clifford + backprop (GCANs) and plain EP + scalars both exist. The unexplored territory is everything else.

---

## 4. Dependencies

```bash
# Core geometry and deep learning
pip install torch torchvision torchaudio
pip install cliffordlayers          # Microsoft Research GCAN (ICML 2023); Clifford for PyTorch
pip install torch-geometric         # PyG: graph neural networks
pip install e3nn                    # E(3)-equivariant baselines
pip install egnn-pytorch            # EGNN baseline (Satorras et al.) for N-body

# Experiment infrastructure
pip install einops                  # Tensor manipulation
pip install wandb                   # Experiment tracking
pip install matplotlib seaborn plotly

# EP: no PyPI package — vendor from reference implementation
# Source: https://github.com/smonsays/equilibrium-propagation (~150 lines core)

# Optional domain-specific
pip install gymnasium               # RL environments
pip install dm-control              # MuJoCo for continuous control (if needed)
```

**Key notes:**
- `cliffordlayers` provides: `CliffordLayer`, `CliffordFourier2d`, batched geometric products for Cl(2,0) and Cl(3,0), grade projections. Use as the Clifford backend.
- Spectral normalization: `torch.nn.utils.spectral_norm` — built into PyTorch. Apply to all weight matrices inside energy functions. User confirmed SN helps EP; quantify this in P1.4 and enable by default if confirmed.
- EP engine: vendor `smonsays/equilibrium-propagation` into `cliffeq/training/ep_engine.py` — ~150 lines, easy to modify for multivector states.

---

## 5. Repository Structure

```
cliffeq/
├── cliffeq/
│   ├── algebra/          # Clifford wrappers: grade projection, norm, reversal, products
│   ├── energy/           # Energy function families (norm, bilinear, graph, Hopfield, EBM)
│   ├── dynamics/         # Update rules: all Axis C variants as DynamicsRule subclasses
│   ├── training/         # EP engine, CHL, FF, PC, TP, ISTA, CD training loops
│   ├── attention/        # Clifford attention / geometric attention modules
│   ├── models/           # Full architectures: flat, GNN, hierarchical, Transformer, Hopfield
│   └── benchmarks/       # Task loaders, metrics, baseline runners
├── experiments/          # One script per PoC: p1_1_baseline_grid.py, pv1_vision.py, etc.
├── results/              # JSON logs and plots per run
├── notebooks/            # Visual exploration, energy landscape, convergence analysis
└── TODO.md
```

---

## 6. Foundation Modules

Build these before any PoC. Do not over-engineer — implement exactly what the first PoC needs, then extend.

### F1: Clifford Algebra Utilities
Thin wrapper on `cliffordlayers` standardizing tensor contract `(batch, nodes, components)`.

- [x] Cl(2,0): 4D. Cl(3,0) grade-2 truncated: 7D. Cl(3,0) full: 8D. Cl(1,3): 16D.
- [x] `grade_project(x, grades)` — zero out all but listed grades
- [x] `reverse(x)` — negate grades 2 and 3
- [x] `clifford_norm_sq(x)` — scalar part of x̃x
- [x] `scalar_part(x)`, `vector_part(x)`, `bivector_part(x)` — grade extraction
- [x] Geometric, inner, and outer products (delegate to `cliffordlayers`)
- [x] `embed_scalar(x)` / `embed_vector(x)` — lift real tensors into multivector grade slots

### F2: Energy Function Base + Spectral Norm

- [x] `EnergyFunction(state, weights) -> scalar` abstract base
- [x] `use_spectral_norm: bool` flag — wraps internal weight matrices with `torch.nn.utils.spectral_norm` when True
- [x] Log largest singular value (effective Lipschitz constant) during training when SN active
- [x] All concrete energy functions inherit this base (see P2.1)

### F3: Update Dynamics Base

- [x] `DynamicsRule.step(x, energy_fn, alpha) -> x_new` abstract interface
- [x] All 7 Axis C rules as concrete subclasses (see P1.2)
- [x] All rules: differentiable w.r.t. `x` (needed for EP gradient via autograd), shape-preserving

### F4: EP Engine Adapter

Wrap any energy function into the EP two-phase loop:
```
Free phase:   x* = relax(E, x_init, n_free steps)
Clamped:      x** = relax(E + β·L(scalar(x), target), x*, n_clamped steps)
ΔW ∝ (1/β)·[∂E(x*)/∂W − ∂E(x**)/∂W]
```

- [x] `EPEngine(energy_fn, dynamics_rule, n_free, n_clamped, beta, dt, use_spectral_norm)`
- [x] `.free_phase(x_init)` / `.clamped_phase(x_init, target)` / `.parameter_update(x_free, x_clamped)`
- [x] Works identically for scalar or Clifford states — the difference is only in the dynamics rule

### F5: Forward-Forward Engine

Hinton's Forward-Forward algorithm adapted for Clifford states:
```
Positive pass: real (x, y) → compute goodness G(h) = ‖h‖² − θ → maximize
Negative pass: fake (x, y') → compute goodness G(h) → minimize
Layer-local update: no global gradient, no backward pass
```

- [x] `FFEngine(goodness_fn, threshold_theta)` — layer-local training loop
- [x] Default goodness: `G(h) = clifford_norm_sq(h) − θ` (Clifford norm instead of ‖·‖²)
- [x] Alternative goodness: `G(h) = scalar_part(h̃ W h) − θ` (learnable geometric goodness)
- [x] Positive/negative data generation: support label-mixing, noise corruption, and adversarial negatives

### F6: Clifford Geometric Attention Module

The key insight: **Modern Hopfield retrieval is one step of EP dynamics with the Hopfield energy.** Replacing the dot product with a Clifford inner product `scalar(Q̃ ✶ K)` gives attention that is aware of orientation, not just magnitude.

```
Standard:    Attention(Q,K,V) = softmax(QK^T / √d) V
Clifford:    GeoAttention(Q,K,V) = softmax(β · scalar(Q̃ ✶ K)) V
             or richer: incorporate bivector(Q̃ ✶ K) as orientation bias
```

- [x] `CliffordAttention(n_heads, clifford_dim, use_orientation_bias: bool)`
- [x] `orientation_bias`: if True, add `bivector(Q̃ ✶ K)` projected to scalar as an additive bias to attention logits — encodes relative orientation between query and key without explicit positional encoding
- [x] Compatible with standard Transformer blocks as a drop-in replacement for `nn.MultiheadAttention`
- [x] When `clifford_dim=1` (scalar only), degenerates to standard dot-product attention

### F7: Metrics & Logging

- [x] `equivariance_violation(model, x, group)` — group ∈ {SO2, SO3, O3, Z2, Sn (permutation)}
- [x] `convergence_curve(energy_fn, dynamics, x_init, n_steps)` → energy per step
- [x] `fixed_point_count(energy_fn, dynamics, n_init=200)` → number of distinct attractors
- [x] `MetricsLogger` — wandb if configured, JSON fallback
- [x] Standardized `run_experiment(config)` harness logging: task metric, equivariance violation, convergence iterations, energy residual at fixed point, wall-clock, peak GPU memory

---

## 7. Non-Backprop Algorithm Reference

Quick-reference for how each training algorithm connects to the Clifford-EP framework.

| Algorithm | Core Idea | Clifford Combination | Priority |
|---|---|---|---|
| **EP** | Free + clamped phases; ΔW = state difference | Clifford states + geometric update rules | Primary |
| **CHL** | Positive + negative phases; same formula as EP | Multivector outer products in ΔW | High |
| **Forward-Forward** | Goodness maximized locally per layer | Clifford norm or geometric goodness | High |
| **Predictive Coding** | Minimize prediction error layer-locally | Multivector predictions; geometric residuals | Medium |
| **Target Propagation** | Compute inversion-based targets per layer | Geometric targets; Clifford inversion via reversal | Medium |
| **ISTA / Sparse Coding** | Soft-threshold energy minimization | Clifford dictionary atoms; graded sparsity | Medium |
| **Contrastive Divergence** | MCMC positive + short-chain negative | Clifford EBM; Langevin in multivector space | Medium |
| **Modern Hopfield** | EP with Hopfield energy = attention | Clifford inner product → geometric attention (F6) | High |

---

## 8. Phase 1: Core Toy Baselines

**Purpose:** Establish whether the mechanism works at all. Minutes to run. Fast iteration. All subsequent experiments reference these results.

---

### P1.1: The Fundamental 2×2 Grid

**Task:** 2D rotation-invariant binary classification: points inside a unit circle vs. inside a rotated ellipse. A perfectly equivariant model generalizes to all rotation angles from a single training orientation.

**The grid (all parameter-matched):**

| | Backprop | Equilibrium Propagation |
|---|---|---|
| Scalar states | MLP-BP ← baseline | Scalar EP (Scellier) ← baseline |
| Clifford Cl(2,0) | Clifford-BP (GCAN-style) ← baseline | **Clifford-EP ← novel** |

Run each × {SN off, SN on} = 8 total configurations.

- [x] All 4 model variants, shared task loader and metrics harness (F7)
- [x] Report all F7 metrics; equivariance violation is the key comparative metric

**Success criterion:** Clifford-EP converges on every run; its equivariance violation < scalar EP's; accuracy ≥ 85%.
**Kill switch:** If Clifford-EP fails to converge → do P2.1 (energy zoo) before proceeding.

---

### P1.2: Update Rule Shootout

**Task:** Same as P1.1. Compare all 7 Axis C dynamics rules. This is the most impactful single design decision.

- [x] All 7 rules from F3; 5 seeds each; 20 iterations max
- [x] Report: convergence curve, final accuracy, equivariance violation, wall-clock/iteration
- [x] Plot: all rules on same energy-vs-iteration graph

**Success criterion:** At least one geometric rule achieves equivariance violation < 1e-3 without convergence regression vs. LinearDot.
**Decision:** Whichever rule wins becomes the default for all subsequent PoCs. If LinearDot wins, note that dynamics matter less than state representation.

---

### P1.3: Grade Truncation Ablation

**Configurations in Cl(3,0):**

| Config | Dim | Notes |
|---|---|---|
| G0 | 1D | Scalar only — degenerate baseline |
| G01 | 4D | Scalar + vector |
| G02 | 4D | Scalar + bivector (no position, only orientation) |
| G012 | 7D | Scalar + vector + bivector — proposed sweet spot |
| G0123 | 8D | Full algebra |

- [x] All 5, same Clifford-EP (best rule from P1.2), same parameter count (pad if needed)
- [x] Report: accuracy, equivariance violation, convergence speed, memory
- [x] Plot: accuracy vs. dimensionality Pareto curve

**Success criterion:** G012 achieves ≥95% of G0123 accuracy at ≤90% wall-clock cost.

---

### P1.4: Spectral Normalization Quantification

- [x] 3 conditions: no SN | SN on scalar EP | SN on Clifford-EP
- [x] Sweep nudge strengths β ∈ {0.01, 0.1, 0.5}
- [x] Track: convergence speed, energy residual oscillation, fixed-point stability
- [x] **Decision:** If SN reduces convergence iterations by >20% or eliminates oscillation → enable by default everywhere.

---

### P1.5: Forward-Forward + Clifford (FF Baseline)

**Goal:** Hinton's FF algorithm is backprop-free, layer-local, and requires no clamped phase. Does it work with Clifford states? How does it compare to Clifford-EP?

- [x] Implement `FFEngine` (F5) with Clifford goodness functions
- [x] Goodness variant A: `G(h) = clifford_norm_sq(h) − θ` (Clifford norm)
- [x] Goodness variant B: `G(h) = scalar_part(h̃ W h) − θ` (learnable geometric goodness)
- [x] Negative data: label-corrupted (standard FF), noise-injected, adversarial (worst-case test)
- [x] Compare: scalar FF, Clifford-FF-A, Clifford-FF-B vs. Clifford-EP (P1.1)
- [x] Report: convergence, equivariance violation, sensitivity to negative data quality

**Novel question:** Does a geometric goodness function (Clifford norm) train more stable FF networks than the standard squared-norm goodness?

---

### P1.6: Algebra Signature Exploration

Test whether the algebra signature matters for non-geometric tasks.

- [x] Cl(2,0): 4D — 2D Euclidean
- [x] Cl(3,0): 7D grade-2 — 3D Euclidean (default)
- [x] Cl(1,3) or Cl(3,1): Minkowski — try on time-series with causal structure
- [x] PGA Cl(3,0,1): projective — motors encode translation + rotation; test on path-planning toy

Task: for each algebra, design a tiny synthetic task that exploits its natural symmetry. If Cl(1,3) doesn't help on temporal tasks, note that and move on.

---

### P1.7: Scalar EBM Baseline (Contrastive Divergence)

**Goal:** Establish the "CD-trained EBM" baseline before making it Clifford. This ensures we understand what CD-trained models look like on the same tasks.

- [x] Standard EBM: `E(x) = f_θ(x)` scalar energy, CD training with short Langevin chains
- [x] Clifford-EBM: `E(x) = scalar_part(x̃ W x)`, CD training with Clifford-Langevin
- [x] **Clifford-Langevin:** `x_{t+1} = x_t − α ∂E/∂x + ε` where ε is Clifford-valued noise (noise per grade)
- [x] Compare against Clifford-EP on P1.1 task: CD vs. EP as training algorithm for the same Clifford energy

---

## 9. Phase 2: Structural Explorations — Novel Combinations

**Purpose:** Qualitatively different architectures. Most creative and most speculative. Run in parallel with Phase 1 as capacity allows.

---

### P2.1: Energy Function Zoo

Different energy families produce fundamentally different fixed-point landscapes. This catalog is essential for knowing what is tractable.

| Name | Formula | Key property |
|---|---|---|
| NormEnergy | `E = ‖x‖² = scalar(x̃x)` | Always converges; no learning signal in dynamics |
| BilinearEnergy | `E = scalar(x̃ W x)` | Primary target; W is learnable |
| GraphEnergy | `E = Σ_{ij} scalar(x̃_i W_ij x_j)` | Pairwise; natural for graphs |
| GradeWeightedEnergy | `E = Σ_k λ_k ‖⟨x⟩_k‖²` | Per-grade eigenvalue spectrum |
| HopfieldEnergy | `E = −log Σ_m exp(β · scalar(ξ̃_m x))` | Modern Hopfield; attention-like |
| AsymmetricEnergy | `E = scalar(x̃ W x)`, W ≠ Wᵀ | Allows directed flows; may not have minima |
| HigherOrderEnergy | `E = scalar((x̃ A x)(x̃ B x))` | Quartic; richer landscape; expensive |
| GradeMixingEnergy | `E = scalar(x̃ W x) + scalar(⟨x⟩₁ V ⟨x⟩₂)` | Cross-grade coupling |
| SparseCliffordEnergy | `E = ½‖y − Ax‖² + Σ_k λ_k ‖⟨x⟩_k‖₁` | Clifford sparse coding |

- [x] Implement all 9 as `EnergyFunction` subclasses with `use_spectral_norm` flag
- [x] For each: characterize fixed-point landscape (attractor count, symmetry of attractors)
- [x] Run on P1.1; note convergence, accuracy, oscillation behavior
- [x] Visualize energy landscape for Cl(2,0) (2D-projectible)

**Output:** Catalog of energy families with convergence properties — reference for all further work.

---

### P2.2: Clifford-Hopfield Memory Network

**Novel hypothesis:** Replacing real-valued Hopfield patterns with Clifford multivectors enables *orientation-equivariant* associative retrieval — a rotated query retrieves the same memory as the unrotated query.

```
Patterns: {ξ₁,...,ξN} ⊂ Cl(3,0)   (learned multivectors)
Energy:   E(x) = −log Σ_m exp(β · scalar(ξ̃_m ✶ x))
Dynamics: x_{t+1} = softmax(β · [scalar(ξ̃_m ✶ x)]_m) · [ξ_m]_m   (EP free phase)
```

Note: this energy IS the Modern Hopfield energy with Clifford inner product. Retrieval dynamics are one step of EP.

- [x] Task: store K oriented 3D shape templates; query with noise or partial occlusion
- [x] **Key test:** query rotated by θ ∈ [0°, 360°] → same pattern retrieved (equivariant retrieval)
- [x] Capacity test: max patterns storable before retrieval degrades vs. dimension
- [x] Compare: scalar Hopfield, Modern Hopfield (softmax), quaternion Hopfield, Clifford-Hopfield
- [x] **Connection to attention:** CliffordHopfield retrieval IS CliffordAttention (F6) with patterns as keys/values

**Success criterion:** Equivariant retrieval accuracy >80% for arbitrary rotation; scalar Hopfield should degrade significantly.

---

### P2.3: Rotor-State EP — Equilibrium on SO(3)

**Novel:** States are unit quaternions (rotors in Cl(3,0) even subalgebra). EP finds a stable rotation, not a stable feature vector. The fixed point IS a geometric transformation.

```
State:    q ∈ H₁ (unit quaternion)
Energy:   E(q) = 1 − scalar(q̃ W q)
Dynamics: Riemannian gradient descent on S³ (project ∇E onto tangent space, then retract)
```

- [x] Rotor normalization after each step: `q ← q / ‖q‖`
- [x] Task A: **rotation regression** — predict 3D object orientation from observation
- [x] Task B: **rotation composition** — given two input rotors, predict their product (tests algebraic structure preservation at fixed point)
- [x] Compare: MLP quaternion output, `e3nn` equivariant baseline, Rotor-EP
- [x] **Landscape analysis:** energy on S³ is well-studied — do attractors form expected symmetry orbits?

---

### P2.4: Geometric Equilibrium GNN (GEN-GNN)

**Novel:** EP is inherently local — each unit updates from local energy gradient only. This maps perfectly onto graph message-passing with Clifford states. The global equilibrium emerges from local geometric consistency, with no global aggregation required.

```
Nodes: x_i ∈ Cl(3,0) grade-2 (7D)
Edges: E_ij = scalar(x̃_i W_ij x_j)
Total: E = Σ_{(i,j)} E_ij
EP:    each x_i updates by ∂E/∂x_i = Σ_j W_ij x_j  (only neighbors)
```

- [x] Implement on `torch-geometric`; directed and undirected edge variants
- [x] Task 1: graph classification (MUTAG)
- [x] Task 2: symmetric lattice regression (synthetic; known rotational symmetry)
- [x] Compare: GCN, GAT, GCAN (Clifford-backprop), GEN-GNN (Clifford-EP)
- [x] **Key question:** does truly local EP produce globally equivariant representations without explicit global pooling?

---

### P2.5: Clifford-ISTA — Geometric Sparse Coding

**Novel:** Replace real-valued dictionary atoms with Clifford multivectors. Sparsity penalty is graded: different λ_k per grade allows, e.g., dense scalar activations + sparse bivector activations.

```
min_x ½‖y − Ax‖² + Σ_k λ_k ‖⟨x⟩_k‖₁    where A has Clifford-valued columns
Update: x^(t+1) = grade_soft_thresh(x^(t) − (1/L) A^T(Ax^(t) − y), {λ_k/L})
```

LISTA variant: learn A, λ_k jointly (unrolled ISTA as network layers, but layers are Clifford).

- [x] Implement `CliffordISTA` and `CliffordLISTA`
- [ ] Task: sparse reconstruction of geometric signals (3D point clouds, optical flow)
- [ ] Compare: standard ISTA, Clifford-ISTA, Clifford-EP
- [ ] **Novel question:** do Clifford dictionary atoms learn geometrically interpretable filters (oriented edges, oriented planes)?

---

### P2.6: Clifford Predictive Coding

**Novel:** Predictions are multivectors; prediction errors are geometric residuals.

```
Layer l predicts layer l−1: x̂_{l−1} = W_l ✶ x_l   (geometric product)
Error:  ε_l = x_l − x̂_l      (multivector residual)
Update: x_l ← x_l − α ε_l;  W_l ← W_l + η (ε_{l−1} ⊗ x_l)
```

- [x] Task: masked reconstruction (MNIST with 50% masked; test on rotated test set)
- [ ] Compare: scalar PC, Clifford-BP autoencoder, Clifford-PC
- [ ] **Key question:** do multivector prediction errors carry orientation information that scalar PC discards, resulting in geometrically consistent reconstructions?

---

### P2.7: Clifford Target Propagation

**Novel:** Compute layer targets as geometric inverses. If forward pass is `x_l = f(x_{l−1})`, the target for layer l−1 given target `x_l^target` is: `x_{l−1}^target ≈ f⁻¹(x_l^target)` where inversion uses the Clifford reversal: `f⁻¹(y) ≈ W̃ ✶ y / ‖W‖²`.

- [x] Task: same as P2.6 (masked reconstruction)
- [ ] Compare: standard TP, Clifford-TP, Clifford-EP, Clifford-PC
- [ ] **Key question:** is geometric inversion via Clifford reversal a better layer-target approximation than pseudo-inverse?

---

### P2.8: Geometric Attention as EP (Clifford Transformer Block)

**Key insight:** Modern Hopfield retrieval IS a single step of EP with Hopfield energy. Therefore, **Clifford attention IS a Clifford-EP module** with Hopfield energy. We can train a Clifford attention block using EP (local Hopfield update) instead of backpropagation.

```
GeoAttention(Q,K,V) = softmax(β · scalar(Q̃ ✶ K)) · V
OrientationBias:       + scalar projection of bivector(Q̃ ✶ K)  [optional]
```

The bivector part of Q̃✶K encodes the relative rotation between query and key — orientation-aware attention without explicit positional encoding.

- [ ] Implement `CliffordAttention` (F6) as a drop-in for `nn.MultiheadAttention`
- [ ] Train with EP (Hopfield energy) vs. backprop
- [ ] Task A: sequence classification with known rotational/permutation symmetry (synthetic)
- [ ] Task B: actual language task — character-level prediction on text8 or enwik8 (small)
- [ ] Compare: standard attention + backprop, Clifford attention + backprop, Clifford attention + EP
- [ ] **Key metric:** does orientation bias improve attention on tasks with geometric structure in the sequence?

---

### P2.9: Hybrid Architecture — Clifford-EP Bottleneck Layer

**The general-purpose test:** If the Clifford-EP idea is truly general, inserting a single Clifford-EP bottleneck layer into any existing architecture should improve its equivariance and generalization without requiring the whole model to be geometric.

**Bottleneck interface:**
```
Input: scalar feature tensor (batch, d)
  ↓ F1.embed_vector: project to multivectors (batch, d/7, 7)
  ↓ Clifford-EP: run n_free iterations to geometric equilibrium
  ↓ F1.scalar_part: extract scalar component (batch, d/7)
Output: scalar feature tensor (batch, d/7)
```

- [x] Implement `CliffordEPBottleneck(n_free, dynamics_rule, energy_fn)` as a standard `nn.Module`
- [ ] Insert into ResNet-18 between layer2 and layer3 → test on CIFAR-10 under rotation
- [ ] Insert into a 2-layer Transformer → test on text classification (SST-2)
- [ ] Insert into actor-critic MLP → test on CartPole with mirror symmetry
- [ ] In each case: compare original model vs. model + bottleneck; no other changes
- [ ] **Success criterion:** bottleneck improves equivariance violation and/or OOD accuracy in at least 2 of 3 domains
- [ ] **This is the "general-purpose" finding** — if confirmed, it motivates a standalone publication

---

### P2.10: Multi-Algorithm Comparison on a Single Task

**Goal:** On one well-chosen task (N-body dynamics, 5 particles), run ALL non-backprop algorithms from Section 7 with Clifford states and compare directly.

| Algorithm | Clifford variant |
|---|---|
| EP | Clifford-EP (best rule from P1.2) |
| CHL | Clifford-CHL |
| FF | Clifford-FF (best goodness from P1.5) |
| PC | Clifford-PC (P2.6) |
| TP | Clifford-TP (P2.7) |
| ISTA | Clifford-ISTA (P2.5) |
| CD | Clifford-EBM with Langevin (P1.7) |
| Backprop | GCAN baseline |

- [ ] All 8 on identical N-body task; same parameter count; 5 seeds
- [ ] This is the definitive "which non-backprop algorithm works best with Clifford states" experiment
- [ ] Expected outcome: EP and FF are strongest (both avoid global gradients); CD may struggle on small-data regime

---

## 10. Phase 3: Domain Benchmarks

Run these after Phase 1–2 have identified the best-performing variants. Use the best dynamics rule (P1.2), best grade config (P1.3), SN on/off per P1.4. Each domain tests a different aspect of the Clifford Advantage Hypothesis.

---

### Vision

**The geometric structure in images:** Edge orientations are bivectors. Spatial positions are vectors. Color channels can be grade-1 (3-vector in RGB space). Patches have translation and rotation symmetry.

#### PV1: Clifford-EP for Rotation-Invariant Image Classification

**Task:** CIFAR-10 (or STL-10) under random SO(2) rotation. Train on upright images; test on randomly rotated. Standard CNNs degrade; equivariant models should not.

- [ ] Input: image patches → multivectors (luminance = grade-0, position = grade-1 vector, edge orientation from Sobel filter = grade-2 bivector)
- [ ] Model: Clifford-EP network with grid topology (pixels = nodes in GEN-GNN)
- [ ] Baselines: standard CNN, `e3nn`-based equivariant CNN, Clifford-CNN (backprop via `cliffordlayers` Fourier layers)
- [ ] Metrics: accuracy on upright test, accuracy on rotated test, SO(2) equivariance violation
- [ ] **Key question:** does Clifford-EP preserve more rotation equivariance than Clifford-backprop through training?

#### PV2: Clifford Fourier Vision + EP

`cliffordlayers` includes `CliffordFourier2d` — a Clifford-valued Fourier layer operating on 2D grids. Combine with EP training.

- [ ] Clifford Fourier layer as the energy function: `E(x) = ‖CliffordFourier(x) − target‖²`
- [ ] EP trains the Clifford Fourier feature extractor without backprop
- [ ] Task: texture classification (DTD dataset or synthetic rotated textures)
- [ ] Compare: standard Fourier features + backprop, Clifford Fourier + backprop, Clifford Fourier + EP
- [ ] **Why this is novel:** Clifford Fourier layers exist; training them with EP does not

#### PV3: Scene Geometry Estimation

Geometric quantities (surface normals, depth, optical flow) ARE multivectors — normals are vectors, oriented surfaces are bivectors, flow fields are grade-1. Clifford-EP should be naturally suited.

- [ ] Task: surface normal estimation from single RGB image (NYU Depth v2, small split)
- [ ] Output: per-pixel normal vector → grade-1 multivector → SE(3)-equivariant
- [ ] Model: Clifford-EP applied to patch features from a frozen pretrained CNN backbone
- [ ] Baselines: MLP regression, standard EP, Clifford-BP
- [ ] **Key metric:** angular error of predicted normals; equivariance under camera rotation

---

### Language

**The geometric structure in language:** Less obvious, but real. Syntactic relations have directed structure (dependency arcs = oriented). Semantic composition is more powerful than addition (king − man + woman ≈ queen suggests geometric-algebra-like structure). Positional information may benefit from rotor encoding rather than sinusoidal encoding.

#### PL1: Clifford Token Embeddings + EP Language Model

**Idea:** Represent tokens as multivectors: scalar part = semantic frequency/importance, vector part = distributional semantic direction, bivector part = syntactic relation context.

- [ ] Task: character-level language modeling (text8 or Penn Treebank)
- [ ] Architecture: token embedding → Clifford multivectors → Clifford-EP layers → scalar extraction → softmax
- [ ] Train with EP: energy over (context, next-token) pairs; EP settles on low-energy completion
- [ ] Compare: LSTM (backprop), Clifford-LSTM (backprop), Clifford-EP-LM
- [ ] Metrics: bits-per-character, perplexity, OOD test (different text domain)

#### PL2: Geometric Attention Transformer

**Idea:** Replace dot-product attention with Clifford geometric attention (F6, P2.8) in a small Transformer. Test whether orientation-aware attention improves on structured language tasks.

- [ ] Small Transformer (2-4 layers, ~1M params) with CliffordAttention heads
- [ ] Task A: GLUE SST-2 (sentiment — tests if attention quality matters)
- [ ] Task B: synthetic compositional reasoning (SCAN or COGS — tests structural generalization)
- [ ] Train with backprop first (cleaner baseline), then with EP (Hopfield update) if stable
- [ ] Compare: standard attention, Clifford attention + backprop, Clifford attention + EP
- [ ] **Key question:** does orientation bias in attention improve compositional generalization?

#### PL3: Energy-Based Clifford Sequence Model (JEPA-style)

**Idea:** JEPA (LeCun): predict future representations in latent space, not raw tokens. Clifford states as latent representations; EP finds the low-energy latent that predicts the future.

- [ ] Architecture: encode context → Clifford multivector latent → EP settles → predict next multivector → decode
- [ ] Training: minimize prediction error in Clifford latent space (not token space)
- [ ] Avoids generative decoder; learns predictive geometric representations
- [ ] Task: next-sentence prediction or masked span prediction (Wikipedia sentences)
- [ ] Compare: standard BERT-like masked LM, JEPA with scalar latents, JEPA with Clifford latents
- [ ] **This is the most speculative language PoC** — run last, only if PL1 and PL2 give positive signals

---

### Reinforcement Learning & Control

**The geometric structure in RL:** State spaces have position (vector), velocity (vector), orientation (bivector/rotor), and angular momentum (bivector). Reward functions often have symmetry (mirrored environments, rotated tasks). Multi-agent scenarios have permutation symmetry.

#### PR1: Continuous Control with Geometric Policy

**Task:** MuJoCo HalfCheetah or Ant (via Gymnasium). These have left-right symmetry. A model with Z₂ equivariance should generalize to mirrored versions zero-shot.

- [ ] State encoding: joint angles + velocities → grade-1+grade-2 multivectors (angles as bivectors, velocities as vectors)
- [ ] Policy: Clifford-EP → action (scalar torque commands)
- [ ] Training: energy-based RL — reward as negative energy perturbation (clamped-phase nudge toward high-reward states)
- [ ] Baselines: MLP (PPO), GCAN (Clifford-BP), scalar EP policy
- [ ] **Key test:** train on standard environment; evaluate zero-shot on Z₂-mirrored variant
- [ ] Metrics: episode reward, zero-shot mirror performance, sample efficiency curve

#### PR2: Multi-Agent Swarm Coordination

**Idea:** Each agent is a GEN-GNN node (P2.4). Agents communicate multivector states to neighbors. The swarm relaxes to a globally stable geometric formation via local EP updates — no central coordinator.

- [ ] Task: 2D formation control (maintain triangle/hexagon formation under perturbation)
- [ ] Agents: 6–12 agents, each with Clifford state; communicate to k nearest neighbors
- [ ] EP free phase = all agents relax simultaneously (fully decentralized)
- [ ] Training: clamped phase nudges formation energy toward target configuration
- [ ] Compare: centralized MLP controller, independent PPO agents, Clifford-EP swarm
- [ ] **Key metric:** formation error under random individual agent perturbations; scales to more agents?

---

### Physics & Geometry

These are the most natural domains for Clifford-EP and serve as existence proofs that the approach works before testing harder domains.

#### PG1: N-Body Dynamics Prediction

**Task:** Predict future positions/velocities of N=5 (then N=20) charged particles under Coulomb forces.

- [ ] State: (position, velocity) → grade-1+grade-1 multivector; angular momentum → grade-2 bivector
- [ ] Model: GEN-GNN (P2.4) over fully-connected particle graph
- [ ] Predict: t+1, t+10, t+100
- [ ] Baselines: MLP, EGNN (`egnn-pytorch`), GCAN, scalar EP
- [ ] **Equivariance test:** train on canonical orientation; test on 1000 random SO(3) rotations
- [ ] **Sample efficiency:** accuracy vs. number of training trajectories

**Success criterion:** Clifford-EP matches EGNN t+10 MSE with ≤50% parameters OR lower equivariance violation.

#### PG2: Symmetric Function Suite (Controlled Equivariance Analysis)

Controlled tasks with known ground-truth symmetry — the primary tool for measuring the Clifford Advantage precisely.

| Task | Symmetry | Output type |
|---|---|---|
| 3D convex hull volume | SO(3)-invariant | Scalar |
| Force field prediction | SO(3)-equivariant | Vector field |
| Discrete symmetry detection | Z₂, Z₃, Z₄ | Class label |
| Time-reversal plausibility | T-symmetry (Cl(1,3)) | Binary |

- [ ] Run all Phase 1–2 model variants on all 4 tasks
- [ ] Produce **equivariance vs. accuracy Pareto curves** per model class
- [ ] This is the definitive "which approach gives the best equivariance/accuracy tradeoff" analysis

#### PG3: 3D Point Cloud Classification (ModelNet10, 4-class)

**Task:** Classify 3D shapes from point clouds. 4-class subset (chair, table, bathtub, monitor) for speed.

- [ ] Input: 3D points → grade-1 multivectors (pure vector part)
- [ ] Model: hierarchical Clifford-EP (local Clifford-EP patches → global Clifford-EP)
- [ ] Baselines: PointNet, DGCNN, GCAN
- [ ] **Rotation generalization:** train on upright, test on arbitrary SO(3) rotation

---

## 11. Phase 4: The Clifford Bottleneck Test (Cross-Domain Validation)

**This is the pivotal general-purpose experiment.** If the Clifford-EP bottleneck (P2.9) improves baseline models across at least two of vision, language, and RL without any other changes, it constitutes evidence for a universal geometric processing primitive.

Run P2.9 (Clifford-EP Bottleneck) inserted into:

| Host architecture | Task | Metric |
|---|---|---|
| ResNet-18 | CIFAR-10 + random rotation | Rotated-test accuracy, SO(2) equivariance |
| 2-layer Transformer | SST-2 sentiment | Accuracy; OOD domain shift |
| PPO actor-critic MLP | CartPole + mirror symmetry | Mirror zero-shot reward |
| 2-layer GCN | MUTAG graph classification | Accuracy |

- [ ] Bottleneck adds ~10% parameters; compare same-parameter baselines without bottleneck
- [ ] Ablation: Clifford-EP bottleneck vs. Clifford-BP bottleneck vs. standard MLP bottleneck of same size
- [ ] **Report:** does the EP training matter, or is Clifford representation sufficient? Does EP add anything over just using Clifford states with backprop?

---

## 12. Analysis & Visualization

Build alongside PoCs as needed.

### A1: Fixed-Point Geometry Visualizer
- [ ] Cl(2,0): animate multivector state during relaxation (rotor as rotating arrow, bivector as shaded sector)
- [ ] Cl(3,0): project to 3D; vector part as point, bivector as oriented plane
- [ ] Side-by-side: scalar EP vs. Clifford-EP convergence trajectories

### A2: Attractor Landscape Analysis
- [ ] Sample 500+ random initializations → converge → cluster fixed points by ‖x_i − x_j‖ < ε
- [ ] Visualize attractor distribution: how many attractors? Are they symmetry-related?
- [ ] **Hypothesis:** attractors form orbits under the energy's symmetry group

### A3: Equivariance Drift Tracker
- [ ] Plot equivariance violation vs. training step for each model
- [ ] Does Clifford-EP maintain equivariance through training, or drift? How does this compare to Clifford-backprop?

### A4: Algorithm × Domain Heatmap
- [ ] 2D matrix: rows = algorithm variants; columns = domain tasks; cells = best metric
- [ ] Annotate: which configurations use SN, which grade truncation, which dynamics rule
- [ ] This is the primary research output summary

### A5: Convergence and Stability Atlas
- [ ] For each energy function (P2.1): plot convergence curves from diverse initializations
- [ ] Identify: which energies reliably converge? Which are sensitive to step size? Which exhibit oscillation?

---

## 13. Exploration Priority Order

Follow this sequence. Stop and drill deeper at any point where results are surprising — surprises (positive or negative) are the most valuable signal.

```
FOUNDATION
  F1–F7: core modules (build incrementally as needed by each PoC)

PHASE 1 — Establish basic viability
  P1.1 → P1.2 → P1.4 → P1.3 → P1.5 (FF) → P1.7 (EBM/CD)

PHASE 2 — Structural exploration (most creative work)
  P2.1 (energy zoo) → P2.2 (Hopfield) → P2.3 (Rotor-EP)
  → P2.4 (GEN-GNN) → P2.8 (geometric attention)
  → P2.9 (bottleneck) ← most important if Phase 1 is positive
  → P2.5 (ISTA) → P2.6 (PC) → P2.7 (TP) → P2.10 (algorithm shootout)

PHASE 3 — Domain benchmarks
  PG1 (N-body) → PG2 (symmetric suite) → PR1 (control) → PV1 (vision)
  → PL1 (language LM) → PL2 (geometric attention) → PG3 (point clouds)
  → PR2 (swarm) → PV2 (Fourier) → PV3 (scene) → PL3 (JEPA)

PHASE 4 — Cross-domain bottleneck test (run after PG1, PV1, PR1 show direction)
  P2.9 bottleneck in ResNet + Transformer + PPO + GCN
```

**Rule:** If any Phase 1 PoC fails unexpectedly, investigate the energy function (P2.1) and grade truncation (P1.3) before assuming the framework is broken. Most failures will be energy-design issues, not fundamental impossibilities.

---

## 14. Kill Switches & Decision Points

| After | Gating question | If YES | If NO |
|---|---|---|---|
| P1.1 | Does Clifford-EP converge? | Continue P1.2 | Do P2.1 first; energy form may be wrong |
| P1.2 | Does any geometric rule outperform LinearDot on equivariance? | Use that rule as default | LinearDot as default; geometric richness lives in energy, not dynamics |
| P1.4 | Does SN improve convergence by >20%? | Enable SN everywhere by default | Optional; flag sensitivity |
| P1.5 | Does Clifford-FF converge and train? | Add FF to all subsequent algorithm comparisons | Keep as minor variant; don't invest further |
| P2.4 | Does GEN-GNN outperform GCN? | Push to PG1, PR2 (graph-based domains) | Revisit energy function; local EP may be insufficient |
| P2.8 | Does CliffordAttention improve over dot-product? | Prioritize PL1, PL2 (language) | Language direction is weak; focus on geometric domains |
| P2.9 | Does the bottleneck improve ≥2 domains? | Core publishable result; expand systematically | Framework may need more than a bottleneck; revisit hybrid architecture |
| PG1 | Does Clifford-EP match EGNN on N-body? | Pursue rigorous benchmarks | Identify gap: energy design? Dynamics? Scale? |
| PL1 | Does Clifford-EP LM converge and not blow up? | Pursue PL2, PL3 | Language direction is the hardest; defer to Phase 4 |

---

## 15. Evaluation Philosophy — What Makes a Finding Worth Pursuing

A PoC result is worth deeper investigation if it shows at least one of:

1. **Equivariance improvement:** Clifford-EP model has equivariance violation < 1/3 of the best non-Clifford baseline on the same task. This demonstrates the geometric representation is doing something real.

2. **Sample efficiency gain:** Clifford-EP reaches 90% of its final accuracy with less than 50% the training data of the best baseline. This shows the geometric prior is informative.

3. **OOD generalization:** Clifford-EP model degrades by less than 10% absolute when tested on rotated/reflected/permuted versions of the task it was trained on, while the best baseline degrades by >20%.

4. **Convergence stability:** Clifford-EP converges reliably (>90% of seeds) while scalar EP with the same architecture has >30% divergence rate. This demonstrates that geometric structure stabilizes energy minimization.

5. **Parameter efficiency:** Clifford-EP achieves the same task metric as the best baseline with <60% of the parameters. The geometric prior is doing the work that would otherwise require more parameters.

A result qualifies as a **negative finding worth reporting** if the Clifford-EP model fails to improve on all five dimensions but is no worse — this suggests the framework has neutral cost (safe to use) even when it doesn't help.

A result is a **fundamental failure** (redirect the approach) only if Clifford-EP is both worse than the scalar EP baseline on task metrics AND fails to converge reliably.

---

## 16. Novel Contribution Framing

Every PoC answers a question that **neither plain Clifford nor plain EP could address alone**. The table below maps each PoC to its unique contribution claim.

| PoC | Unique claim |
|---|---|
| P1.1 | Clifford states and EP training are compatible at all |
| P1.2 | The correct geometric iteration rule (no prior theory predicts this) |
| P1.5 | A geometric goodness function trains better FF networks than scalar norm |
| P2.2 | Multivector Hopfield memories support orientation-equivariant retrieval |
| P2.3 | EP can find fixed points on the SO(3) manifold (all prior EP is Euclidean) |
| P2.5 | Graded sparsity in Clifford space learns geometrically interpretable dictionary atoms |
| P2.8 | Clifford inner product as attention score encodes relative orientation without positional encoding |
| P2.9 | A single Clifford-EP bottleneck improves equivariance and OOD performance in arbitrary architectures |
| P2.10 | Which non-backprop algorithm is best suited for geometric state training |
| PL2 | Geometric attention improves compositional generalization in language |
| PR2 | Decentralized local EP dynamics produce stable global geometric formations |
| Phase 4 | Clifford-EP is domain-agnostic: the bottleneck helps across vision, language, and RL |

---

## 17. Open Research Questions

These are the unknowns the framework must resolve.

1. **EP gradient validity for multivectors.** EP's theoretical guarantee (free–clamped difference ≈ parameter gradient) was proved for scalar states. Does it hold for non-commutative Clifford states? This is an open theoretical question; P1.1 and P1.2 test it empirically.

2. **Energy vs. dynamics: which matters more?** Does geometric structure in the energy function provide most of the benefit, with dynamics being secondary? Or does the geometric update rule matter independently? P2.1 + P1.2 together answer this.

3. **Are Clifford fixed points geometrically interpretable?** When the network settles, does the multivector state have readable geometric content (vector part points toward target, bivector encodes orientation), or is it an arbitrary Clifford-format representation? A2 addresses this.

4. **Equivariance through training.** The architecture is equivariant at initialization. Do EP weight updates preserve equivariance over training, or does optimization break it? Does SN help preserve it? A3 addresses this.

5. **The language question.** Is there any form of geometric structure in language that Clifford multivectors can exploit? Or is language the domain where this framework doesn't help? PL1 and PL2 answer this.

6. **Scaling.** Can Clifford-EP scale past toy tasks without becoming computationally prohibitive? The bottleneck architecture (P2.9) is the primary scaling strategy; Phase 4 tests it.
