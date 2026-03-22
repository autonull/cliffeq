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
| **CGA Cl(4,1), 32D** | **Conformal: translations + rotations as unified motors; natural for vision + robotics** |
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

Vision · Language · Reinforcement Learning · Physics/Geometry · Graphs · Scientific simulation · **Molecular/Chemical**

**The novel research is in non-trivial intersections of all four axes.** Plain Clifford + backprop (GCANs) and plain EP + scalars both exist. The unexplored territory is everything else.

---

## 4. Dependencies

```bash
# Core geometry and deep learning
pip install torch torchvision torchaudio
pip install cliffordlayers          # Microsoft Research GCAN (ICML 2023); Cl(2,0) and Cl(3,0) products
pip install torch-geometric         # PyG: graph neural networks + QM9, ModelNet10, MUTAG built-in
pip install e3nn                    # E(3)-equivariant baselines (NequIP uses this)
pip install egnn-pytorch            # EGNN baseline (Satorras et al. 2021, ICML)

# CGA Cl(4,1) — cliffordlayers does NOT support Cl(4,1)
pip install clifford                # Pure-Python general Clifford algebra; use clifford.Cl(4,1)
                                    # for CGA experiments (P1.6 CGA item, PM2)

# Experiment infrastructure
pip install einops wandb
pip install matplotlib seaborn plotly
pip install datasets                # HuggingFace datasets: SST-2, SCAN, WikiText-103

# RL
pip install gymnasium[mujoco]       # MuJoCo environments (HalfCheetah-v4, Ant-v4, CartPole-v1)
pip install stable-baselines3       # PPO implementation; extend ActorCriticPolicy for bottleneck

# NLP tokenization
pip install transformers            # AutoTokenizer for SST-2; BERT tokenizer

# Molecular / QM9 baselines
pip install nequip                  # NequIP: E(3)-equivariant interatomic potential (Batzner et al.)
# SchNet and DimeNet++ are in torch_geometric.nn.models — no separate install needed

# Vision equivariance baselines
pip install e2cnn                   # Group-equivariant CNNs (Weiler & Cesa 2019); p4/p8 rotation groups

# EP: no PyPI package — vendor from reference implementation
# Source: https://github.com/smonsays/equilibrium-propagation (~150 lines core)
# Already vendored into cliffeq/training/ep_engine.py
```

**Key notes:**
- `cliffordlayers` provides `CliffordLayer`, `CliffordFourier2d`, batched products for Cl(2,0) and Cl(3,0) only. For CGA Cl(4,1), use the `clifford` package: `import clifford; layout, blades = clifford.Cl(4, 1)`.
- Spectral normalization: `torch.nn.utils.spectral_norm` — built into PyTorch. Apply to all weight matrices inside energy functions. Quantify in P1.4; enable by default if SN reduces convergence iterations >20%.
- EP engine: already in `cliffeq/training/ep_engine.py`; works for scalar and Clifford states — the state type is determined by the dynamics rule and energy function.
- `torch_geometric` includes QM9, ModelNet10, MUTAG, TUDataset out of the box — no separate download scripts needed.
- text8 dataset: `wget http://mattmahoney.net/dc/text8.zip && unzip text8.zip` — 100M chars, 27-char vocab.

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
  - **Results:** Clifford-FF-B (51.6%) > Clifford-FF-A (48.0%) > Scalar FF (47.6%) on P1.1 task.

**Novel question:** Does a geometric goodness function (Clifford norm) train more stable FF networks than the standard squared-norm goodness?

---

### P1.6: Algebra Signature Exploration

Test whether the algebra signature matters for non-geometric tasks.

- [x] Cl(2,0): 4D — 2D Euclidean
- [x] Cl(3,0): 7D grade-2 — 3D Euclidean (default)
- [x] Cl(1,3) or Cl(3,1): Minkowski — try on time-series with causal structure
- [x] PGA Cl(3,0,1): projective — motors encode translation + rotation; test on path-planning toy
- [x] **CGA Cl(4,1): 32D even subalgebra (16D motors) — conformal model; translations + rotations as unified versors; test on rigid-body motion prediction**
  - Basis: {e₁,e₂,e₃,e₊,e₋} with e₊²=+1, e₋²=−1; null vectors eₒ=½(e₋−e₊), e∞=e₊+e₋
  - Conformal point embedding: `X = x + ½|x|²e∞ + eₒ` for 3D point x
  - Motor = translation × rotation: `M = T·R` where `T = 1 + ½t·e∞`, `R = cos(θ/2) + sin(θ/2)·B̂`
  - EP state: even subalgebra of Cl(4,1), 16D; normalize after each step: `M ← M / ‖M‖`
  - Task: predict rigid-body trajectory (4-atom tetrahedron, random forces/torques; 500 synthetic trajectories, 50-step each); target = motor at t+5 from t=0..4
  - Compare: Cl(3,0)-EP (position only, 7D), PGA-EP (5D motor), CGA-EP (16D motor)
  - Implementation: use `clifford` package — `import clifford; layout, blades = clifford.Cl(4, 1)` — to implement products; wrap as `CliffordState` with custom `geometric_product` override in `cliffeq/algebra/utils.py`
  - Key question: does CGA's unified translation+rotation algebra improve trajectory prediction vs. handling them separately?

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
- [x] **Task A — Oriented image patches:** extract 8×8 patches from CIFAR-10 (32×32 → 16 patches per image); encode each patch as Cl(3,0) multivector: luminance=grade-0, centroid position=grade-1 (normalized to [−1,1]), dominant Sobel edge orientation=grade-2 bivector (compute Sobel Gx,Gy → angle θ → bivector via `embed_bivector_angle(θ)`) → (batch, 16, 7); dictionary=64 Clifford atoms, λ={grade-0:0.01, grade-1:0.05, grade-2:0.1} (sparser on orientation); train LISTA on reconstruction; metric: MSE vs. sparsity (ℓ₀/dim) tradeoff curve
- [ ] **Task B — Optical flow:** MPI Sintel dataset clean pass (Dosovitskiy et al. 2015), 4 short sequences for speed; flow vector at each pixel = grade-1 multivector (vx,vy=grade-1 components); magnitude=grade-0; curl/divergence of local flow window=grade-2; dictionary=128 atoms; compare Clifford-ISTA atoms vs. real-valued DCT and PCA atoms on reconstruction MSE
- [x] **Compare:** (i) standard ISTA (real-valued atoms, L2 reconstruction), (ii) Clifford-ISTA (graded sparsity, Clifford atoms), (iii) Clifford-EP (BilinearEnergy, same dimension) — metric: reconstruction error vs. sparsity tradeoff; sparsity = mean fraction of zero code components
- [x] **Visualization:** for each learned Clifford atom, render vector part as arrow and bivector part as oriented plane sector; success = ≥30% of atoms have ‖grade-1 part‖ > 0.5‖full atom‖, i.e., directional structure; standard ISTA atoms should not show this pattern
- [ ] **Novel question:** do Clifford dictionary atoms learn geometrically interpretable filters (oriented edges, oriented planes) — reminiscent of V1 simple cells?

---

### P2.6: Clifford Predictive Coding

**Novel:** Predictions are multivectors; prediction errors are geometric residuals.

```
Layer l predicts layer l−1: x̂_{l−1} = W_l ✶ x_l   (geometric product)
Error:  ε_l = x_l − x̂_l      (multivector residual)
Update: x_l ← x_l − α ε_l;  W_l ← W_l + η (ε_{l−1} ⊗ x_l)
```

- [x] Task: masked reconstruction — MNIST 28×28, 50% random pixel mask; train/test standard 60k/10k split; also evaluate on rotated test set (rotate each test image by 15°, 30°, 45°, 90° using `torchvision.transforms.functional.rotate`); input encoding: pixel intensity=grade-0, (px,py) position normalized to [−1,1]=grade-1; masked pixels zeroed out
- [x] **Compare:** (i) scalar PC — real-valued prediction errors, real-valued weight updates (Rao & Ballard 1999 architecture: 2 layers, 128→64 hidden); (ii) Clifford-BP autoencoder — encoder-decoder CNN with same parameter count, trained end-to-end; (iii) Clifford-PC (as described); metric: reconstruction MSE on unmasked pixels, SSIM, equivariance violation on rotated test set (F7)
- [x] **Key question:** does the grade-2 (bivector) component of prediction error `ε_l` correlate with local edge orientation in the masked region? Test: compute Sobel orientation in ground-truth unmasked image at masked pixels; correlate with bivector magnitude of `ε_l` at those positions. If correlation > 0.3, PC errors carry orientation information.
- [ ] **Reference:** Rao & Ballard 1999 "Predictive coding in the visual cortex" (Nat. Neurosci.); Millidge et al. 2022 "Predictive Coding: Towards a Future of Deep Learning" (arXiv)

---

### P2.7: Clifford Target Propagation

**Novel:** Compute layer targets as geometric inverses. If forward pass is `x_l = f(x_{l−1})`, the target for layer l−1 given target `x_l^target` is: `x_{l−1}^target ≈ f⁻¹(x_l^target)` where inversion uses the Clifford reversal: `f⁻¹(y) ≈ W̃ ✶ y / ‖W‖²`.

- [x] Task: same as P2.6 (MNIST masked reconstruction; same encoding, same train/test splits)
- [x] **Compare:** (i) DTP — difference target propagation (Lee et al. 2015, NIPS; reference: `github.com/theonegoodboy/difference-target-propagation`); (ii) Clifford-TP (reversal inversion); (iii) Clifford-EP; (iv) Clifford-PC from P2.6 — all parameter-matched; metric: reconstruction MSE, layer-target alignment distance `‖x_l^target − x_l^true‖` per layer averaged over test set
- [x] **Clifford reversal inversion implementation detail:** given forward pass `x_l = σ(W_l ✶ x_{l-1})`, compute target for layer l−1 as `x_{l−1}^target ≈ W̃_l ✶ σ⁻¹(x_l^target) / ‖W_l‖²` where W̃_l is the Clifford reverse of W_l (negate components of grade 2 and 3); compare this against standard pseudo-inverse `W_l⁺ σ⁻¹(x_l^target)`; track alignment error per layer
- [x] **Key question:** is Clifford reversal a better layer-inversion approximation than pseudo-inverse? Measured by: mean ‖x_l^target − x_l^true‖ for each layer (lower = more accurate target), averaged across test set and 5 seeds
- [ ] **Reference:** Lee et al. 2015 "Difference Target Propagation" (NIPS); Ernoult et al. 2022 "Towards Biologically Plausible and Private Gene Expression Data Generation" (discusses TP variants)

---

### P2.8: Geometric Attention as EP (Clifford Transformer Block)

**Key insight:** Modern Hopfield retrieval IS a single step of EP with Hopfield energy. Therefore, **Clifford attention IS a Clifford-EP module** with Hopfield energy. We can train a Clifford attention block using EP (local Hopfield update) instead of backpropagation.

```
GeoAttention(Q,K,V) = softmax(β · scalar(Q̃ ✶ K)) · V
OrientationBias:       + scalar projection of bivector(Q̃ ✶ K)  [optional]
```

The bivector part of Q̃✶K encodes the relative rotation between query and key — orientation-aware attention without explicit positional encoding.

- [x] **Wire `CliffordAttention` (already in `cliffeq/attention/geometric.py`) as a drop-in for `nn.MultiheadAttention`:** match the signature `forward(query, key, value, attn_mask=None, key_padding_mask=None) → (output, attn_weights)`; Q/K/V projected from token embeddings via linear → reshape to `(batch, seq, n_heads, clifford_dim=8)`; score: `s_ij = scalar(Q̃_i ✶ K_j) / sqrt(d_k)`; orientation bias: add `α · scalar_part(bivector(Q̃_i ✶ K_j))` to logits where α is a learnable per-head scalar; output = weighted sum of V_j → extract scalar_part for downstream; verify `clifford_dim=1` degenerates to standard dot-product attention
- [x] **EP training for the attention block:** model the attention block as a Hopfield energy `E(x) = −log Σ_j exp(β · scalar(Q̃ ✶ K_j))`; free phase = run T_free=3 attention iterations (re-compute Q from current x, re-attend); for supervised tasks clamp the output [CLS] token to target class embedding; weight update via standard EP formula; compare against backprop-trained identical architecture
- [x] **Task A — Synthetic rotational sequence** (fast validation): 16 tokens × 8D features; class = dominant rotation angle {0°, 90°, 180°, 270°}; 2000 train / 500 test; rotated sequences should map to same class (permutation equivariance); use `generate_synthetic_sequence_data` already in `experiments/p2_8_geometric_attention.py`; key metric: accuracy on circularly-shifted versions of training sequences
  - **Results:** Clifford Attention (50.0%) > Standard Attention (46.0%). EP-trained Clifford Attention viable (43.0%).
- [ ] **Task B — text8 character language model** (main result): 100M-char Wikipedia corpus, 27-char vocab; download `http://mattmahoney.net/dc/text8.zip`; split 90M/5M/5M train/val/test; seq_len=256, batch=64; architecture: 4-layer Transformer, 256 hidden, 4 heads with CliffordAttention, FFN dim=512, ~2M params; metric: bits-per-character (bpc = cross_entropy_nats / ln(2)); target range: LSTM ≈1.43 bpc, small Transformer ≈1.35 bpc (Merity et al. 2018)
- [ ] **Compare (4 variants, parameter-matched):** (i) standard `nn.MultiheadAttention` + backprop, (ii) `CliffordAttention` (no orientation bias) + backprop, (iii) `CliffordAttention` + orientation bias + backprop, (iv) `CliffordAttention` + EP (Hopfield update)
- [ ] **Key metrics:** (a) does orientation bias (variant iii) improve over no-bias (variant ii)? (b) does EP training (variant iv) match backprop bpc? (c) on Task A: does CliffordAttention improve permutation equivariance vs. standard attention?
- [ ] **Reference:** Ramsauer et al. 2021 "Hopfield Networks is All You Need" (ICLR) — proves Modern Hopfield = attention; Merity et al. 2018 AWD-LSTM text8 bpc baseline; Katharopoulos et al. 2020 "Transformers are RNNs" (ICML)

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

- [x] Implement `CliffordEPBottleneck` — V3 in `cliffeq/models/bottleneck_v3_proper.py`; unit-tested (31 tests pass); maintains dimensionality (no compression), gradient flows correctly
- [ ] **Domain 1 — Vision:** ResNet-18 (`torchvision.models.resnet18(weights=None)`, 11.2M params) + CIFAR-10 under random rotation
  - Insert after `layer2` (output: batch×128×8×8): flatten spatial → `(batch, 64, 128)` → bottleneck (128D Clifford state, n_free=5) → reshape back → `layer3`
  - CIFAR-10 rotation: train on clean images; test on two sets: (i) clean, (ii) each image rotated by angle drawn uniformly from [0°,360°] at test time; `torchvision.transforms.functional.rotate`
  - Bottleneck params ≤1.1M (10% of ResNet-18); verify with `sum(p.numel() for p in bottleneck.parameters())`
  - Metrics: clean accuracy, rotated accuracy, SO(2) equivariance violation (F7), params, wall-clock
- [ ] **Domain 2 — Language:** 2-layer Transformer (256 hidden, 4 heads, ~1M params) + SST-2 sentiment + SCAN compositional split
  - SST-2: `datasets.load_dataset("glue", "sst2")`; tokenizer: `AutoTokenizer.from_pretrained("bert-base-uncased")`; max_len=128; [CLS] pooling
  - Insert bottleneck after FFN sublayer of Transformer block 1; input dim=256 → (batch, seq, 256) → group 256 into 32 Clifford states → bottleneck → restore dim
  - OOD test: train on SST-2, evaluate on SemEval-2017 Task 4A (Twitter sentiment, `datasets.load_dataset("sem_eval_2018_task_1", "subtask5.english")`) for domain shift
  - SCAN length split: `datasets.load_dataset("scan", "length")`; ~17k train, 4.2k test; exact sequence accuracy
  - Metrics: SST-2 val accuracy, SemEval OOD accuracy, SCAN exact-match accuracy
- [ ] **Domain 3 — RL:** PPO actor-critic (`stable-baselines3.PPO`) + CartPole-v1 + Z₂ mirror
  - Actor MLP: (4→64→64→2); insert bottleneck after layer 1 (64D → bottleneck with n_free=3 → 64D)
  - Z₂ mirror: `mirrored_obs = obs * mirror_mask` where `mirror_mask = [-1, -1, 1, 1]` for CartPole (negate cart position and velocity; keep pole angle and angular velocity); evaluate 100 mirrored episodes without retraining
  - Custom SB3 policy: subclass `stable_baselines3.common.policies.ActorCriticPolicy`, override `mlp_extractor` to insert bottleneck; PPO hyperparams: n_steps=2048, batch=64, lr=3e-4, n_epochs=10
  - Metrics: mean reward (standard, 100 episodes), mean reward (mirrored, zero-shot), timesteps to 475-reward threshold
- [ ] **Domain 4 — Graphs:** 2-layer GCN + MUTAG
  - `torch_geometric.datasets.TUDataset(root='.', name='MUTAG')` — 188 graphs, binary; standard 10-fold CV (stratified)
  - Insert bottleneck between GCN layer 1 (7→64) and layer 2 (64→64): node features (batch×N, 64) → bottleneck (n_free=5, 64D Clifford) → GCN layer 2
  - Metrics: mean test accuracy ± std across 10 folds
- [ ] **Domain 5 — Molecular:** SchNet (`torch_geometric.nn.models.SchNet`) + QM9 U₀ target
  - `torch_geometric.datasets.QM9(root='data/QM9')`; split: 110k/10k/13.8k (Schütt et al. 2018 convention); target index 7 (U₀, eV); report MAE in meV
  - Insert bottleneck after last SchNet interaction block, before output network; node feature dim=128 → bottleneck (n_free=5) → output MLP
  - Metrics: U₀ MAE (meV), SO(3) equivariance violation
- [ ] **Ablation (all 5 domains, 4 variants each):** (i) original model, (ii) + MLP bottleneck (same param count, ReLU, no Clifford), (iii) + Clifford-BP bottleneck (Clifford structure, end-to-end backprop), (iv) + Clifford-EP bottleneck; this separates: does adding params help? does Clifford structure help? does EP help over Clifford+backprop?
- [ ] **Success criterion:** Clifford-EP bottleneck improves equivariance violation and/or OOD metric vs. original model in ≥2 of 5 domains AND outperforms Clifford-BP bottleneck in ≥1 domain (to show EP is doing something beyond just Clifford structure)
- [ ] **This is the "general-purpose" finding** — if confirmed across 2+ domains, this motivates a standalone publication

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

- [ ] **Dataset:** 5-particle Coulomb N-body system; 3000 train / 500 val / 500 test trajectories; each trajectory 500 steps; task = predict position at t+5 from states t=0..4; use same generator as PG1 (`generate_nbody_system` in `p3_1_clifford_ep_nbody.py`)
- [ ] **Parameter matching:** each algorithm gets exactly 50k trainable parameters; verify with `sum(p.numel() for p in model.parameters())`; adjust hidden dims: EP≈64 hidden, FF≈48 hidden (extra goodness layer), ISTA≈128-atom dictionary, etc.
- [ ] **8 variants — implementation pointers:**
  - EP: `EPEngine` in `cliffeq/training/ep_engine.py`; best dynamics rule from P1.2; `BilinearEnergy`; n_free=20, β=0.1
  - CHL: `CHLEngine` in `cliffeq/training/chl_engine.py`; positive phase=forward data, negative phase=phase-reversed data; same energy as EP
  - FF: `FFEngine` in `cliffeq/training/ff_engine.py`; best goodness function from P1.5; layer-local, no clamped phase
  - PC: Clifford-PC from P2.6; 3 layers; geometric product predictions; layer-local weight updates
  - TP: Clifford-TP from P2.7; Clifford reversal inversion; 3 layers
  - ISTA: `CliffordISTA` from `cliffeq/models/sparse.py`; 64-atom dictionary; λ={grade-0:0.01, grade-1:0.05, grade-2:0.1}; 20 unrolled iterations
  - CD: `CDEngine` in `cliffeq/training/cd_engine.py`; 10-step Langevin MCMC; step size=0.01; Clifford-valued noise per grade
  - Backprop (GCAN): `CliffordLayer` from `cliffordlayers`; standard Adam; same depth and width
- [ ] 5 seeds each; report mean ± std: test MSE, equivariance violation, wall-clock time/epoch, convergence curve (MSE vs. epoch up to 100 epochs)
- [ ] **Output:** Algorithm × Metric heatmap table — feeds directly into Analysis A4; expected: EP and FF strongest; ISTA suboptimal (N-body states are not sparse); CD struggles (MCMC in 7D Clifford space is expensive)
- [ ] **Reference:** Scellier & Bengio 2017 "Equilibrium Propagation" (Front. Comput. Neurosci.); Hinton 2022 "The Forward-Forward Algorithm" (arXiv); Friston 2005 "A theory of cortical responses" (Phil. Trans. R. Soc.) for PC

---

## 10. Phase 3: Domain Benchmarks

Run these after Phase 1–2 have identified the best-performing variants. Use the best dynamics rule (P1.2), best grade config (P1.3), SN on/off per P1.4. Each domain tests a different aspect of the Clifford Advantage Hypothesis.

---

### Vision

**The geometric structure in images:** Edge orientations are bivectors. Spatial positions are vectors. Color channels can be grade-1 (3-vector in RGB space). Patches have translation and rotation symmetry.

#### PV1: Clifford-EP for Rotation-Invariant Image Classification

**Task:** CIFAR-10 (or STL-10) under random SO(2) rotation. Train on upright images; test on randomly rotated. Standard CNNs degrade; equivariant models should not.

- [ ] **Dataset:** CIFAR-10 (50k/10k); train on upright; two test splits: (i) clean, (ii) randomly rotated (uniform angle ∈ [0°,360°] per image, `torchvision.transforms.functional.rotate`); for faster iteration also test on RotMNIST (60k/10k, same rotation protocol)
- [ ] **Input encoding:** divide each 32×32 image into 16 non-overlapping 8×8 patches; per patch: grade-0 = mean luminance; grade-1 = (cx, cy, 0) centroid normalized to [−1,1]; grade-2 = dominant Sobel edge bivector: compute Gx,Gy → angle θ = atan2(Gy,Gx) → bivector components (cos2θ, sin2θ, 0) in the (e₁∧e₂, e₁∧e₃, e₂∧e₃) slots; result: (batch, 16, 7) grade-2-truncated Cl(3,0) state
- [ ] **Model:** GEN-GNN (P2.4) on 4-nearest-neighbor patch graph (by spatial proximity); 3 EP free-phase steps; `BilinearEnergy`; readout: scalar_part → linear → 10-class softmax; ~500k params
- [ ] **Baselines:**
  - Standard CNN: 3× (Conv3×3 → BN → ReLU) + global average pool + linear; ~500k params
  - Group-equivariant CNN (p4): `from e2cnn.gspaces import Rot2dOnR2; gspace = Rot2dOnR2(4)`; 4-fold rotation equivariance; `pip install e2cnn`; Cohen & Welling 2016 (ICML)
  - Equivariant CNN (e3nn): lift patches to 3D spherical harmonics; `e3nn.o3.Irreps("1x0e + 1x1o + 1x2e")`; strongest equivariant baseline
  - Clifford-CNN (backprop): same GEN-GNN topology + same patch encoding, but replace EP with backprop-trained `CliffordLayer` from `cliffordlayers`
- [ ] **Metrics:** accuracy (clean), accuracy (rotated), SO(2) equivariance violation (F7), params, wall-clock per epoch
- [ ] **Reference:** Cohen & Welling 2016 "Group Equivariant CNNs" (ICML); Weiler & Cesa 2019 "General E(2)-Equivariant Steerable CNNs" (NeurIPS); Ruhe et al. 2023 "Clifford Group Equivariant Neural Networks" (NeurIPS)

#### PV2: Clifford Fourier Vision + EP

`cliffordlayers` includes `CliffordFourier2d` — a Clifford-valued Fourier layer operating on 2D grids. Combine with EP training.

- [ ] **Dataset:** DTD (Describable Textures Dataset; Cimpoi et al. 2014, CVPR) — 47 texture classes, 120 train + 120 val + 120 test images per class (~5640 images total); `torchvision.datasets.DTD(root, split='train'/'val'/'test', download=True)`; resize to 128×128; evaluate with 10 random train/test splits (DTD protocol)
- [ ] **`CliffordFourier2d` usage:** `from cliffordlayers.models.utils.clifford_fourier import CliffordFourier2d`; signature `[1,1]` for Cl(2,0) or `[1,1,1]` for Cl(3,0); apply to (batch, C, H, W) Clifford feature maps; use as energy function: `E(x) = ‖CliffordFourier(x) − CliffordFourier(x_ref)‖²` where x_ref is a class exemplar; EP drives x toward low-energy Clifford-frequency match
- [ ] **Alternatively:** use `CliffordFourier2d` as a learnable feature extractor; EP trains the extractor weights by minimizing inter-class energy separation (positive/negative pairs); this avoids needing a class exemplar at test time
- [ ] **Compare:** (i) Gabor filter bank + SVM, (ii) CNN features + backprop, (iii) Clifford Fourier features + backprop, (iv) Clifford Fourier features + EP; all evaluated on same DTD 10-split protocol; metric: mean accuracy ± std across 10 splits
- [ ] **Reference:** Cimpoi et al. 2014 "Describing Textures in the Wild" (CVPR); Brandstetter et al. 2023 "Clifford Neural Layers for PDE Modeling" (ICLR) for CliffordFourier2d

#### PV3: Scene Geometry Estimation

Geometric quantities (surface normals, depth, optical flow) ARE multivectors — normals are vectors, oriented surfaces are bivectors, flow fields are grade-1. Clifford-EP should be naturally suited.

- [ ] **Dataset:** NYU Depth v2 (Silberman et al. 2012) — 1449 RGBD images, 795 train / 654 test; surface normals computed from depth maps via cross-product of horizontal and vertical gradients; install via `pip install gdown` and download from `https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html`; resize to 160×120 for speed
- [ ] **Input:** frozen ResNet-50 backbone (pretrained ImageNet, `torchvision.models.resnet50(weights='DEFAULT')`); extract layer3 features (batch, 1024, 5, 4) at 160×120; patchify → embed as Cl(3,0): luminance=grade-0, (px,py)=grade-1, Sobel gradient=grade-2; (batch×20, 1024→7 projected) Clifford states
- [ ] **Output:** per-pixel surface normal → decode grade-1 part of equilibrium state → L2-normalize → (nx, ny, nz); use upsampling head (bilinear) to restore spatial resolution
- [ ] **Metrics:** mean angle error (degrees) = mean arccos(n̂_pred · n̂_gt); standard thresholds: % within 11.25°, 22.5°, 30°; SOTA (deep network) ≈ 14° mean angle error; equivariance under 90° camera rotation (rotate image → predicted normals should rotate consistently)
- [ ] **Baselines:** (i) MLP regression directly from ResNet features, (ii) standard scalar EP on same features, (iii) Clifford-BP (same arch, backprop); reference: Wang et al. 2015 "Designing Deep Networks for Surface Normal Estimation" (CVPR), mean error ≈ 19°

---

### Language

**The geometric structure in language:** Less obvious, but real. Syntactic relations have directed structure (dependency arcs = oriented). Semantic composition is more powerful than addition (king − man + woman ≈ queen suggests geometric-algebra-like structure). Positional information may benefit from rotor encoding rather than sinusoidal encoding.

#### PL1: Clifford Token Embeddings + EP Language Model

**Idea:** Represent tokens as multivectors: scalar part = semantic frequency/importance, vector part = distributional semantic direction, bivector part = syntactic relation context.

- [ ] **Dataset:** text8 (Mahoney 2009) — 100M Wikipedia characters, 27-char vocab (a-z + space); `wget http://mattmahoney.net/dc/text8.zip && unzip text8.zip`; split: 90M train / 5M val / 5M test; seq_len=256; batch=128; for faster iteration also try PTB word-level (`torchtext.datasets.PennTreebank`)
- [ ] **Architecture (~500k params total):** character → learned scalar embedding (dim=64) → `embed_vector` (F1) → Cl(3,0) multivector (batch, seq, 8); 2 Clifford-EP layers (BilinearEnergy, LinearDot, n_free=5); readout: scalar_part → linear → 27-class softmax; causal masking: energy only couples position i to positions j < i (lower-triangular W_ij)
- [ ] **EP training for autoregressive LM:** free phase: all positions except last relax with causal energy; clamped phase: clamp last position to one-hot target embedding and relax again; weight update via standard EP formula; note: autoregressive EP is a new variant — document carefully
- [ ] **Compare (all parameter-matched to ~500k):**
  - LSTM: `nn.LSTM(64, 256, 2, batch_first=True)` + linear head; target ≈1.43 bpc on text8 (Merity et al. 2018)
  - Clifford-LSTM: LSTM hidden state is Cl(3,0) multivector; gates via scalar_part; trained with backprop
  - Clifford-EP-LM: described above; metric: bpc = cross_entropy_nats / ln(2)
- [ ] **OOD test:** evaluate text8-trained model on PTB test set (bpc); compare degradation across model types; expect Clifford representations to transfer better if they learn structural patterns
- [ ] **Reference:** Merity et al. 2018 "Regularizing and Optimizing LSTM Language Models" (ICLR) for AWD-LSTM text8 baseline; Mikolov et al. 2012 "Subword Language Modeling" for text8 benchmark

#### PL2: Geometric Attention Transformer

**Idea:** Replace dot-product attention with Clifford geometric attention (F6, P2.8) in a small Transformer. Test whether orientation-aware attention improves on structured language tasks.

- [ ] **Architecture:** 4-layer Transformer (256 hidden, 4 CliffordAttention heads, FFN dim=512, ~4M params); replace `nn.MultiheadAttention` with `CliffordAttention` from P2.8; orientation bias: learnable per-layer scalar α
- [ ] **Task A — SST-2:** `datasets.load_dataset("glue", "sst2")`; `AutoTokenizer.from_pretrained("bert-base-uncased")`; max_len=128; [CLS] pooling → 2-class head; 5 seeds; target: ≥88% validation accuracy with 4M-param model; this is a sanity check — if Clifford attention can't match standard attention here, something is wrong
- [ ] **Task B — SCAN compositional generalization (the critical test):** `datasets.load_dataset("scan", "length")` — train on short commands (max 22 actions), test on long commands (48–500 actions); 16728 train / 4182 test; sequence-to-sequence task: command string → action string; metric: exact sequence accuracy on length split; standard Transformer ≈17%, LSTM ≈14%, SCAN SOTA compositional models ≈95%; orientation bias in attention is hypothesized to help by encoding compositional structure — test this
  - Also run: `datasets.load_dataset("scan", "addprim_jump")` — generalize 'jump' primitive; standard Transformer ≈1%; harder test of compositionality
- [ ] **Train with backprop first** on both tasks; if stable, add EP training for attention block (Hopfield update, n_free=3); EP may improve on SCAN where structural consistency matters
- [ ] **Compare:** (i) standard `nn.MultiheadAttention` + backprop, (ii) `CliffordAttention` (no bias) + backprop, (iii) `CliffordAttention` + orientation bias + backprop, (iv) `CliffordAttention` + EP; all same param count
- [ ] **Reference:** Lake & Baroni 2018 "Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence RNNs" (ICML); Ontanon et al. 2022 "Making Transformers Solve Compositional Tasks" (ACL); Ramsauer et al. 2021 "Hopfield Networks is All You Need" (ICLR)

#### PL3: Energy-Based Clifford Sequence Model (JEPA-style)

**Idea:** JEPA (LeCun): predict future representations in latent space, not raw tokens. Clifford states as latent representations; EP finds the low-energy latent that predicts the future.

- [ ] **Architecture (I-JEPA style, Assran et al. 2023 CVPR):** context encoder (4-layer Transformer, 256 hidden) → Clifford multivector latent ([CLS] token, 8D Cl(3,0)); predictor (2-layer Clifford-EP, n_free=5) → predicted next-sentence latent; EMA target encoder (momentum=0.996, updated each step) to prevent collapse; training signal: `L = ‖latent_predicted − stopgrad(latent_target)‖²` in Clifford space (mean over grades)
- [ ] **Dataset:** WikiText-103 (Merity et al. 2016) — 103M word tokens; `torchtext.datasets.WikiText103`; split consecutive sentence pairs (context → next sentence); ~1.8M train pairs / 3761 val / 4358 test
- [ ] **Evaluation:** fine-tune [CLS] latent representations on SST-2 and MNLI (1k labeled examples each) after pretraining; compare transfer accuracy vs. pretraining method; this tests whether Clifford predictive latents are better sentence representations
- [ ] **Compare:** (i) BERT-style masked LM (predict tokens, not latents; standard BERT small), (ii) JEPA with scalar latents (same architecture, real-valued [CLS]), (iii) JEPA with Clifford latents
- [ ] **This is the most speculative language PoC** — run last, only if PL1 and PL2 give positive signals on equivariance or compositional tasks
- [ ] **Reference:** LeCun 2022 "A Path Towards Autonomous Machine Intelligence" (arXiv); Assran et al. 2023 "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR); Devlin et al. 2019 BERT (NAACL)

---

### Reinforcement Learning & Control

**The geometric structure in RL:** State spaces have position (vector), velocity (vector), orientation (bivector/rotor), and angular momentum (bivector). Reward functions often have symmetry (mirrored environments, rotated tasks). Multi-agent scenarios have permutation symmetry.

#### PR1: Continuous Control with Geometric Policy

**Task:** MuJoCo HalfCheetah or Ant (via Gymnasium). These have left-right symmetry. A model with Z₂ equivariance should generalize to mirrored versions zero-shot.

- [ ] **Environment:** `gymnasium.make("HalfCheetah-v4")`; 17D state (8 joint positions + 9 joint velocities), 6D action (torques); `pip install gymnasium[mujoco]`; Z₂ mirror = multiply state by mask `[1, −1, −1, −1, −1, −1, −1, −1, 1, 1, −1, −1, −1, −1, −1, −1, −1]` and negate actions `[−1, −1, −1, −1, −1, −1]` (Ordonez-Apraez et al. 2023 exact HalfCheetah mask); also try CartPole-v1 as cheaper first test (4D state, Z₂ mirror = negate positions/velocities = `[−1,−1,1,1]`)
- [ ] **State encoding:** joint angles → grade-2 bivectors via `embed_bivector_angle(θ_i)` in e₁∧e₂ plane; joint velocities → grade-1 vectors; pack per-joint as Cl(3,0) multivector → GEN-GNN over joint connectivity graph (HalfCheetah joints: hip→thigh→shin×2 legs); or simpler: flatten → embed as sequence of Clifford multivectors for MLP-style policy
- [ ] **Simplest viable approach first:** use standard PPO with a Clifford-EP bottleneck (from P2.9 Domain 3) inserted into actor and critic MLPs; only after confirming this works, attempt full Clifford-EP policy
- [ ] **Training:** PPO via `stable_baselines3.PPO`; custom policy by subclassing `ActorCriticPolicy`; hyperparams: n_steps=2048, batch=64, n_epochs=10, lr=3e-4, clip_range=0.2, gamma=0.99; total timesteps: 1M (CartPole), 3M (HalfCheetah)
- [ ] **Mirror zero-shot test:** after training, evaluate 100 episodes with mirrored observations (apply Z₂ mask); zero-shot = no fine-tuning; a Z₂-equivariant policy achieves same reward on mirrored env as standard env
- [ ] **Baselines:** (i) MLP PPO (SB3 default), (ii) GCAN policy (Clifford layers + backprop), (iii) scalar EP policy; all ~same param count
- [ ] **Metrics:** mean episode reward (standard), mean episode reward (mirrored zero-shot), timesteps to 475-reward threshold (CartPole), timesteps to 3000-reward threshold (HalfCheetah)
- [ ] **Reference:** Schulman et al. 2017 "Proximal Policy Optimization" (arXiv); Ordonez-Apraez et al. 2023 "MorphoSymmetries" for exact mirror mask specifications; Mondal et al. 2022 "EqR: Equivariant Representations for Data-Efficient RL" (ICML)

#### PR2: Multi-Agent Swarm Coordination

**Idea:** Each agent is a GEN-GNN node (P2.4). Agents communicate multivector states to neighbors. The swarm relaxes to a globally stable geometric formation via local EP updates — no central coordinator.

- [ ] **Environment:** custom 2D formation control — implement in pure Python/NumPy, no simulator needed; 12 agents at 2D positions; target formation = regular hexagon with center agent; formation energy: `E_form = Σ_i ‖pos_i − pos_target_i‖²`; perturbation: kick 2 random agents by ±0.5 unit per step; 1000-step episodes; done if formation MSE < 0.01 or > 100 steps without improvement
- [ ] **Agent state:** (x, y, vx, vy) → Cl(2,0) multivector: grade-0=speed, grade-1=(x,y), grade-2=(vx∧vy) angular momentum; encode as (batch, 4, 4) state tensor; k=3 nearest-neighbor communication graph updated each step
- [ ] **EP swarm dynamics:** all agents simultaneously update `x_i ← x_i − α ∂E_local/∂x_i` where `E_local = Σ_{j∈N(i)} scalar(x̃_i W x_j)` (only neighbor energies); this is one EP free-phase step; T_free=10 steps per decision; shared GEN-GNN weights across all agents (decentralized weight sharing)
- [ ] **Training:** EP clamped phase: clamp all agents to target formation positions; weight update via EP formula; train on 500-episode batches
- [ ] **Compare:** (i) centralized MLP controller (sees all 12 agent states simultaneously), (ii) independent PPO agents (each sees only own state + k neighbor states, no shared training), (iii) Clifford-EP swarm (GEN-GNN, local EP communication)
- [ ] **Scale test:** train on 6 agents; evaluate zero-shot on 12, 24, 48 agents; measures compositionality of learned Clifford representations; expect Clifford-EP swarm to generalize better due to permutation-equivariant GEN-GNN
- [ ] **Metrics:** formation MSE under perturbation; recovery time (steps to formation error < 0.1); performance vs. number of agents (scalability curve)
- [ ] **Reference:** Olfati-Saber 2006 "Flocking for Multi-Agent Dynamic Systems" (IEEE TAC); Tolstaya et al. 2020 "Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks" (CoRL)

---

### Physics & Geometry

These are the most natural domains for Clifford-EP and serve as existence proofs that the approach works before testing harder domains.

#### PG1: N-Body Dynamics Prediction

**Task:** Predict future positions/velocities of N=5 (then N=20) charged particles under Coulomb forces.

- [ ] **Dataset:** use the Satorras et al. 2021 (EGNN) N-body benchmark — 5 charged particles, 3000/2000/2000 train/val/test trajectories, each 1000 timesteps; task = predict position at t=1000 from t=0..499; download data generator from `github.com/vgsatorras/egnn` → `n_body_system/` OR use `generate_nbody_system` in `experiments/p3_1_clifford_ep_nbody.py` (already implements Coulomb forces); also generate a 20-particle version by scaling up
- [ ] **State encoding:** per particle — grade-0: charge scalar; grade-1: 3D position (x,y,z); grade-1 (second): 3D velocity (vx,vy,vz); grade-2: angular momentum L = r × v as 3D bivector (L_xy, L_xz, L_yz); pack as (batch, n_particles, 8) full Cl(3,0) — note: packing two grade-1 vectors requires using Cl(6,0) with 64D or concatenating grade-1 slots; simplest: use grade-0=charge, grade-1=position, grade-2=velocity-bivector (skew encode velocity as bivector via `v ∧ e_ref`)
- [ ] **Model:** GEN-GNN (P2.4) over fully-connected particle graph; edge energy `E_ij = scalar(x̃_i W_ij x_j)` where W_ij depends on Coulomb coupling `q_i q_j / ‖r_i − r_j‖`; 3 EP iterations; readout = grade-1 part of equilibrium state = predicted position
- [ ] **Predict t+1, t+10, t+100** by unrolling the EP dynamics (autoregressive); report MSE for each horizon
- [ ] **Baselines:**
  - MLP: 3-layer, 256 hidden; flattened (position, velocity) input (Satorras 2021 standard MLP)
  - EGNN: `from egnn_pytorch import EGNN_Network`; 4 layers, 64 hidden, equivariant normalization; authors' hyperparams
  - GCAN: `CliffordLayer` from `cliffordlayers` + backprop; same depth/width as GEN-GNN
  - Scalar EP: same GEN-GNN topology but real-valued states
- [ ] **Equivariance test:** apply 1000 random SO(3) rotations R to all particle positions; measure `‖f(Rx) − Rf(x)‖ / ‖f(x)‖`; equivariant models should yield ~0; report mean ± std
- [ ] **Sample efficiency:** train on {100, 300, 1000, 3000} trajectories; plot t+10 MSE vs. training set size; expect Clifford-EP to have flatter degradation curve (geometric prior reduces data need)
- [ ] **Reference:** Satorras, Hoogeboom, Welling 2021 "E(n) Equivariant Graph Neural Networks" (ICML); Brandstetter et al. 2022 "Geometric and Physical Quantities improve E(3) Equivariant Message Passing" (ICLR)

**Success criterion:** Clifford-EP matches EGNN t+10 MSE with ≤50% parameters OR equivariance violation lower than any non-explicitly-equivariant baseline.

#### PG2: Symmetric Function Suite (Controlled Equivariance Analysis)

Controlled tasks with known ground-truth symmetry — the primary diagnostic for measuring the Clifford Advantage precisely.

| Task | Symmetry | Output type |
|---|---|---|
| 3D convex hull volume | SO(3)-invariant | Scalar |
| Force field prediction | SO(3)-equivariant | Vector field |
| Discrete symmetry detection | Z₂, Z₃, Z₄ | Class label |
| Time-reversal plausibility | T-symmetry (Cl(1,3)) | Binary |

- [x] **Implement 4 synthetic tasks:**
  1. [x] **Convex Hull Volume** (SO(3)-invariant): 20 random 3D points → grade-1 multivectors; predict scalar volume; ground truth via `scipy.spatial.ConvexHull(points).volume`; 10k samples, 8k/1k/1k split; metric: MAE; equivariance test: rotate input → output unchanged
  2. [x] **Force Field Prediction** (SO(3)-equivariant): 10 point charges with positions and charge values; predict 3D Coulomb force vector at a query point; ground truth: `F = Σ_i q_i (r_query − r_i) / ‖r_query − r_i‖³`; 10k samples; metric: vector MAE; equivariance test: rotate input → force rotates by same R
  3. [x] **Discrete Symmetry Detection**: 12 2D points; binary label = pattern has Z_n symmetry (n ∈ {2,3,4}); 3 subtasks × 2k samples each; metric: binary accuracy; equivariance test: rotate pattern → prediction unchanged; generate symmetric patterns by applying n-fold rotation to a base configuration
  4. [x] **Time-Reversal Plausibility** (uses Cl(1,3)): 5 spacetime events (t,x,y,z); binary = entropy-consistent (particles moving apart after collision) vs. time-reversed (particles spontaneously converging); 2k samples; Cl(1,3) models should distinguish these via causal metric g=diag(+1,−1,−1,−1); metric: accuracy; this is the one task where Minkowski signature is natural
- [ ] Run all Phase 1–2 model variants on all 4 tasks: scalar EP, Clifford-EP Cl(3,0), Clifford-EP CGA, scalar BP, Clifford-BP Cl(3,0), EGNN (task 2 only)
- [ ] **Produce equivariance vs. accuracy Pareto curves:** x-axis = equivariance violation (lower better, log scale), y-axis = task metric (higher better); one point per model variant; draw Pareto frontier; this is the key figure for a paper showing the geometric advantage
- [ ] This is the definitive "which approach gives the best equivariance/accuracy tradeoff" analysis

#### PG3: 3D Point Cloud Classification (ModelNet10, 4-class)

**Task:** Classify 3D shapes from point clouds. 4-class subset (chair, table, bathtub, monitor) for speed.

- [ ] **Dataset:** `torch_geometric.datasets.ModelNet(root='data/ModelNet', name='10', train=True/False, pre_transform=T.SamplePoints(1024))`; filter to 4 classes: chair (889 train/100 test), table (392/100), bathtub (106/50), monitor (465/100) = ~1852 train / ~350 test; normalize points to unit sphere
- [ ] **Input encoding:** 3D point (x,y,z) → grade-1 slot of Cl(3,0) multivector; optionally add surface normal (estimated via local PCA over k=10 neighbors) as grade-2 bivector via `n̂ = e₁∧e₂ · R(n̂)` (dual construction); result: (batch, 1024, 8) per cloud
- [ ] **Hierarchical model:** Layer 1: k-NN graph k=20 over all 1024 points → GEN-GNN (P2.4), 3 EP steps → local feature multivectors; Layer 2: farthest-point sample to 128 points, k-NN k=10 → GEN-GNN, 3 EP steps → region features; Layer 3: global GEN-GNN over 128 region nodes → single aggregate multivector; readout: scalar_part → 4-class softmax; ~500k params total
- [ ] **Baselines:** PointNet (Qi et al. 2017, CVPR; re-implement or use `torch_geometric.nn.PointNetConv`), DGCNN (`torch_geometric.nn.DynamicEdgeConv`), GCAN (same hierarchical structure but EP replaced by backprop-trained layers); all ~500k params
- [ ] **Rotation generalization:** train on upright (ModelNet default); test on: (i) clean, (ii) SO(3)-randomly rotated (all test instances × 5 random rotations), (iii) SO(2)-rotated (elevation axis only); equivariance violation via F7; expect Clifford-EP to show lower SO(3) violation than DGCNN
- [ ] **Reference:** Qi et al. 2017 "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (CVPR); Wang et al. 2019 "Dynamic Graph CNN for Learning on Point Clouds" (ACM TOG)

---

### Molecular & Chemical

**The geometric structure:** Atoms have 3D positions (grade-1 vectors). Bonds have orientations (grade-2 bivectors). Molecular symmetry groups (point groups) are subgroups of SO(3). This is the domain where SO(3)-equivariance has the most established value and the strongest baselines to compete with.

#### PM1: Molecular Property Prediction (QM9)

**Task:** Predict scalar quantum-chemical properties (atomization energy, HOMO-LUMO gap, dipole moment, etc.) from 3D molecular geometry. QM9 contains ~134k small organic molecules with DFT-computed ground-truth properties.

- [x] **Dataset:** `from torch_geometric.datasets import QM9; dataset = QM9(root='data/QM9')` — 133,885 molecules; 19 DFT properties; standard split: 110k train / 10k val / 13.8k test (Schütt et al. 2018); normalize each property by mean/std of training set
  - Priority targets: U₀ = internal energy at 0K (index 7, units eV, SOTA MAE ≈ 0.009 eV with NequIP); μ = dipole moment (index 0, Debye); HOMO (index 2), LUMO (index 3) energy gaps (eV)
- [ ] **State encoding per atom:** grade-0 = atom type embedding (5 types H,C,N,O,F → linear projection to scalar); grade-1 = 3D position (x,y,z); grade-2 = sum of bond direction bivectors to neighbors: `Σ_{j∈N(i)} (r_j−r_i)/‖r_j−r_i‖ ∧ ê_ref` normalized; result: (batch_nodes, 8) Cl(3,0), with `torch_geometric` batch indexing for variable-size molecules
- [ ] **Model:** GEN-GNN (P2.4) with Gaussian radial basis functions (RBF) of interatomic distance as edge weights (same as SchNet); `W_ij(d) = Σ_k c_k exp(−(d−μ_k)²/σ²)` with K=20 RBF centers on [0,5]Å; 3 message-passing / EP iterations; readout: sum over atoms of `scalar_part(x_i)` → linear → predicted property
- [ ] **Baselines:**
  - SchNet: `torch_geometric.nn.models.SchNet(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50)`; U₀ MAE ≈ 14 meV
  - EGNN: `from egnn_pytorch import EGNN_Network`; 4 layers, 128 hidden; U₀ MAE ≈ 14 meV
  - NequIP: `pip install nequip`; run with default config on QM9; U₀ MAE ≈ 9 meV (Batzner et al. 2022, Nat. Comm.); this is the gold-standard equivariant baseline
  - DimeNet++: `torch_geometric.nn.models.DimenetPlusPlus`; U₀ MAE ≈ 6 meV; uses bond angles (stronger inductive bias)
  - GCAN: Clifford layers from `cliffordlayers` + backprop; same depth as GEN-GNN
- [ ] **Equivariance test:** apply 1000 random SO(3) rotations to all molecular positions; predicted scalar properties (U₀, HOMO, LUMO) should be invariant; report max deviation
- [ ] **Sample efficiency:** train on subsets {1k, 10k, 50k, 110k}; plot U₀ MAE vs. training set size; expect Clifford-EP to have better low-data performance than SchNet/EGNN due to geometric prior
- [ ] **Reference:** Ramakrishnan et al. 2014 "Quantum chemistry structures and properties of 134 kilo molecules" (Sci. Data); Schütt et al. 2018 "SchNet" (NeurIPS); Batzner et al. 2022 "E(3)-equivariant GNNs for data-efficient interatomic potentials" (Nat. Comm.)

**Why this domain is high-value:**
- Largest established SO(3)-equivariance benchmark in ML
- Direct comparison to NequIP/SEGNN which use E(3)-equivariant networks
- If Clifford-EP matches NequIP with fewer parameters → strong publication case
- CGA (Cl(4,1)) may be superior here: molecular conformational changes involve both rotation and translation

**Success criterion:** Clifford-EP within 20% of NequIP MAE on U₀ target, using ≤50% of its parameters. Or: lower equivariance violation than any non-explicitly-equivariant baseline.

#### PM2: CGA Molecular Dynamics (Rigid-Body Trajectory Prediction)

**Task:** Predict rigid-body motions of small molecules (translation + rotation). This is the task CGA was designed for — a single motor encodes the full rigid-body transformation.

- [x] **Prerequisite:** CGA Cl(4,1) from P1.6 must be implemented; use `clifford` package for Cl(4,1) products: `import clifford; layout, blades = clifford.Cl(4, 1); e1,e2,e3,ep,em = blades['e1'],blades['e2'],blades['e3'],blades['e4'],blades['e5']`; eₒ = 0.5*(em−ep), e∞ = ep+em
- [x] **Dataset:** synthetic rigid tetrahedron dynamics — 1000 trajectories; 4-atom rigid body with random initial orientation and position; force = gravity + random perturbation torque; dt=0.01, 100 steps per trajectory; generate with: `pos_next = R(dt)·pos + t(dt)` where R=rotation matrix from torque, t=translation from force; encode each state as CGA motor M = T·R; 800/100/100 train/val/test
- [x] **State encoding:** rigid-body state as even-subalgebra Cl(4,1) motor, 16D (8 independent components after normalization constraint); normalize: `M ← M / ‖M‖` after each EP step; EP energy: `E(M) = 1 − scalar(M̃ W M)` where W is a learnable Cl(4,1) operator
- [x] **Compare (parameter-matched, ~30k params each):**
  - Cl(3,0)-EP: encode (position, orientation quaternion) separately; 7D + 4D = 11D state; BilinearEnergy; n_free=10
  - CGA-EP: single 16D motor state; BilinearEnergy on even subalgebra; n_free=10
  - SE(3)-Transformer (`pip install se3_transformer`; Fuchs et al. 2020 NeurIPS): gold-standard SE(3)-equivariant baseline
  - MLP baseline: flatten (position 3D + quaternion 4D) per atom × 4 atoms = 28D; 3-layer MLP
- [x] **Metrics:** rigid-body MSE = position MSE + quaternion geodesic distance; equivariance violation under SE(3) = combined rotation + translation; comparison of Cl(3,0) vs CGA motor MSE
- [x] **Key question:** does CGA's unified translation+rotation (single motor) improve trajectory prediction vs. separate handling? Measure: CGA-EP rigid-body MSE vs. Cl(3,0)-EP (pos_MSE + orient_MSE)
- [ ] **Reference:** Dorst, Fontijne, Mann 2007 "Geometric Algebra for Computer Science" (Morgan Kaufmann); Fuchs et al. 2020 "SE(3)-Transformers" (NeurIPS); Valkenburg & Dorst 2011 "Estimating Motors from a Variety of Geometric Data in 3D CGA"

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
| **GNN on QM9** | **Molecular property prediction (U₀)** | **MAE; SO(3) equivariance violation** |

- [ ] **Exact insertion points (see P2.9 for full details per domain):** ResNet-18 after layer2; Transformer after FFN of block 1; PPO actor after layer 1; GCN after layer 1; SchNet after last interaction block
- [ ] **Parameter budget:** bottleneck ≤10% of host model params; confirm before each run with `sum(p.numel() for p in bottleneck.parameters())`
- [ ] **4-variant ablation across all 5 domains:** (i) original model, (ii) + MLP bottleneck (same size, no Clifford), (iii) + Clifford-BP bottleneck (Clifford structure, end-to-end backprop), (iv) + Clifford-EP bottleneck; this 5×4 matrix answers: (a) does adding params help? (b) does Clifford structure help beyond params? (c) does EP help over Clifford+backprop?
- [ ] **Report the 5×4 matrix** with key metric per domain + equivariance violation; annotate which variant is best per domain; this is the primary Phase 4 output figure
- [ ] **Success criterion:** Clifford-EP bottleneck improves equivariance violation and/or OOD metric vs. original model in ≥2 of 5 domains AND outperforms Clifford-BP bottleneck in ≥1 domain
- [ ] **If success in ≥3 domains:** this is the core claim for a standalone publication — "A universal geometric inference primitive via Clifford-EP bottleneck layers"
- [ ] **If success in 1 domain:** identify which — likely molecular/physics; reframe as domain-specific contribution rather than universal

---

## 12. Analysis & Visualization

Build alongside PoCs as needed.

### A1: Fixed-Point Geometry Visualizer
- [ ] **Cl(2,0) animation:** at each EP iteration, render 4D multivector state as: grade-0=background brightness, grade-1=(x,y) arrow from origin, grade-2=oriented sector (filled arc in direction of bivector axis); use `matplotlib.animation.FuncAnimation`; save as GIF; show scalar EP vs. Clifford-EP side-by-side; 10 random initializations
- [ ] **Cl(3,0) 3D projection:** use `plotly.graph_objects.Scatter3d`; vector part (grade-1) as 3D point; bivector part (grade-2) as oriented plane disk (construct plane from e₁∧e₂, e₁∧e₃, e₂∧e₃ components); animate relaxation trajectory over EP steps; save as HTML interactive plot
- [ ] **Side-by-side convergence:** for P1.1 task, show 5 random initializations converging to fixed points; color-code by which fixed point they reach; expect Clifford attractors to have geometric symmetry that scalar attractors lack

### A2: Attractor Landscape Analysis
- [ ] Sample 500 random Cl(3,0) initializations → run EP to convergence (energy change < 1e-6); cluster by `‖x_i − x_j‖ < ε=0.1`; report: (a) number of distinct attractor clusters, (b) mean intra-cluster distance, (c) distance between cluster centroids
- [ ] **Symmetry orbit test:** for each attractor cluster, apply all symmetry transformations of the energy (e.g., SO(3) rotations for BilinearEnergy with rotation-invariant W); do rotated attractors map to other attractors in the same cluster? This tests whether attractors form symmetry orbits — a key theoretical prediction
- [ ] Visualize attractor distribution in PCA-projected 2D space; color by cluster; expect structured, not random distribution
- [ ] **Hypothesis:** attractors form orbits under the energy's symmetry group; deviations from this indicate symmetry-breaking by the weight matrix W

### A3: Equivariance Drift Tracker
- [ ] During training (each epoch), compute F7 equivariance violation for: scalar EP, Clifford-EP, Clifford-BP, scalar BP; plot all 4 on same graph (y: violation log scale, x: epoch)
- [ ] **Key question:** does EP training preserve equivariance better than backprop? Does Clifford-EP maintain near-zero violation throughout training, while Clifford-BP drifts up?
- [ ] Also track: singular value spectrum of W (measure of symmetry preservation); `torch.linalg.svdvals(W)` at each epoch; expect SN-enabled models to have flatter spectrum

### A4: Algorithm × Domain Heatmap
- [ ] 2D matrix: rows = 8 algorithm variants (EP, CHL, FF, PC, TP, ISTA, CD, BP); columns = domain tasks (P1.1 toy, PG1 N-body, PV1 vision, PL1 language, PR1 RL, PM1 molecular); cells = best metric (normalized to [0,1] per column)
- [ ] Second layer: annotate each cell with (SN on/off, best dynamics rule, grade config) — the 3-tuple hyperparameter recommendation for each (algorithm, domain) pair
- [ ] Produce with `seaborn.heatmap` or `plotly.imshow`; use diverging colormap (green=above baseline, red=below); this is the primary research output summary figure for a paper

### A5: Convergence and Stability Atlas
- [ ] For each of the 9 energy functions from P2.1 (NormEnergy, BilinearEnergy, GraphEnergy, GradeWeightedEnergy, HopfieldEnergy, AsymmetricEnergy, HigherOrderEnergy, GradeMixingEnergy, SparseCliffordEnergy): plot energy vs. EP iteration from 20 diverse initializations (5 initializations × 4 step sizes α ∈ {0.001, 0.01, 0.1, 1.0})
- [ ] Classify each energy: (i) always converges, (ii) converges for α < α_max (find α_max), (iii) oscillates, (iv) diverges; produce a table
- [ ] **Step-size sensitivity:** for each energy, report α_max (largest stable step size) and n_converge (median iterations to ‖∂E/∂x‖ < 1e-4); this is a practical reference for hyperparameter selection in all subsequent experiments

---

## 13. Exploration Priority Order

Follow this sequence. Stop and drill deeper at any point where results are surprising — surprises (positive or negative) are the most valuable signal.

```
FOUNDATION
  F1–F7: core modules — CONFIRMED CORRECT (67/67 unit tests pass, 2026-03-21)

PHASE 1 — Establish basic viability  [P1.1 done; P1.2–P1.7 implemented, need experimental re-run]
  P1.1 ✓ → P1.2 → P1.4 → P1.3 → P1.5 (FF) → P1.6 (signatures + CGA) → P1.7 (EBM/CD)

PHASE 2 — Structural exploration (most creative work)  [all need experimental run]
  P2.1 (energy zoo) → P2.2 (Hopfield) → P2.3 (Rotor-EP)
  → P2.4 (GEN-GNN) → P2.8 (geometric attention)
  → P2.9 (bottleneck) ← most important if Phase 1 is positive
  → P2.5 (ISTA) → P2.6 (PC) → P2.7 (TP) → P2.10 (algorithm shootout)

PHASE 3 — Domain benchmarks  [all need experimental run; run breadth-first before going deep]
  PG1 (N-body) → PG2 (symmetric suite) → PM1 (QM9 molecular) → PR1 (control)
  → PV1 (vision rotation) → PL1 (language LM) → PL2 (geometric attention)
  → PG3 (point clouds) → PM2 (CGA rigid-body) → PR2 (swarm)
  → PV2 (Fourier) → PV3 (scene) → PL3 (JEPA)

PHASE 4 — Cross-domain bottleneck test (run after PG1, PV1, PR1, PM1 show direction)
  P2.9 bottleneck in ResNet + Transformer + PPO + GCN + GNN/QM9
```

**Rule:** If any Phase 1 PoC fails unexpectedly, investigate the energy function (P2.1) and grade truncation (P1.3) before assuming the framework is broken. Most failures will be energy-design issues, not fundamental impossibilities.

---

## 14. Kill Switches & Decision Points

| After | Gating question | If YES | If NO |
|---|---|---|---|
| P1.1 | Does Clifford-EP converge on all seeds? | Continue P1.2 | Do P2.1 (energy zoo) before proceeding — energy form likely wrong |
| P1.2 | Does any geometric rule (GeomProduct, ExpMap, RotorOnly) outperform LinearDot on equivariance? | Use that rule as default for all experiments | LinearDot as default; note that dynamics matter less than state representation |
| P1.4 | Does SN reduce convergence iterations by >20% or eliminate oscillation? | Enable SN everywhere by default | Make SN optional; keep flag but don't force |
| P1.5 | Does Clifford-FF converge and train (loss decreasing, accuracy >chance)? | Add FF to all subsequent algorithm comparisons (P2.10, Phase 3) | Keep as minor variant; report negative result |
| P1.6 CGA | Does CGA-EP outperform Cl(3,0)-EP on the rigid-body task? | Invest in CGA implementation for PM2, PR1; add CGA column to all Phase 3 tables | Note CGA overhead not worth it; drop PM2 as low-priority |
| P2.4 | Does GEN-GNN outperform GCN on graph classification? | GEN-GNN is primary model for PG1, PM1, PR2 | Investigate energy design; local EP may need better edge energy |
| P2.8 | Does CliffordAttention (orientation bias variant) improve accuracy or equivariance vs. standard attention? | Prioritize PL2 (SCAN compositional), add orientation bias to all Transformer experiments | Language direction is weaker; skip PL3, focus Phase 3 on geometric domains |
| P2.9 | Does bottleneck improve ≥2 of 5 domains? | Core publishable result; write Phase 4 paper | Investigate: is it the Clifford structure or EP that's missing? Run Clifford-BP bottleneck ablation to separate |
| PG1 | Does Clifford-EP match EGNN t+10 MSE (within 20%) or show lower equivariance violation? | Pursue PM1 (QM9) and PG2 as primary physics direction | Identify gap: energy design (try HopfieldEnergy, GradeMixing)? Or need EGNN-style equivariant coordinates? |
| PM1 | Does Clifford-EP U₀ MAE come within 50% of NequIP? | Molecular direction is high-value; invest in CGA-EP (PM2) and full QM9 benchmark | Molecular geometric priors require explicit equivariance (not just Clifford structure); this domain needs fundamentally different approach |
| PL1 | Does Clifford-EP LM train without blowing up and achieve bpc within 20% of LSTM? | Pursue PL2 (geometric attention), PL3 (JEPA) | Language direction is the hardest; deprioritize PL3; keep PL2 only if P2.8 positive |
| Phase 4 | Does bottleneck improve ≥3 of 5 domains? | "Universal geometric primitive" — standalone publication | Identify which domains are positive; frame as domain-specific (physics/molecular) contribution |

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
| P1.6 CGA | CGA motors (unified translation+rotation) improve rigid-body trajectory prediction vs. Cl(3,0) |
| P2.2 | Multivector Hopfield memories support orientation-equivariant associative retrieval |
| P2.3 | EP can find fixed points on the SO(3) manifold (all prior EP is Euclidean) |
| P2.5 | Graded sparsity in Clifford space learns geometrically interpretable dictionary atoms (V1-like) |
| P2.8 | Clifford inner product as attention score encodes relative orientation without positional encoding; improves compositional generalization |
| P2.9 | A single Clifford-EP bottleneck improves equivariance and OOD performance in arbitrary architectures across domains |
| P2.10 | Which non-backprop algorithm is best suited for geometric state training (definitive algorithm comparison) |
| PG2 | Equivariance vs. accuracy Pareto curves across all model variants — the key comparative figure |
| PM1 | Clifford-EP for molecular property prediction on QM9 — competitive with NequIP at fewer parameters |
| PM2 | CGA motors unify translation+rotation as single algebraic object; benefits rigid-body molecular dynamics |
| PL2 | Geometric orientation bias in attention improves compositional generalization on SCAN length split |
| PR2 | Decentralized local EP dynamics produce stable global geometric formations; scales to unseen swarm sizes |
| Phase 4 | Clifford-EP is domain-agnostic: bottleneck helps across 5 diverse domains (vision, language, RL, graphs, molecular) |

---

## 17. Open Research Questions

These are the unknowns the framework must resolve.

1. **EP gradient validity for multivectors.** EP's theoretical guarantee (free–clamped difference ≈ parameter gradient) was proved for scalar states with symmetric energy. Does it hold for non-commutative Clifford states where W may not be self-adjoint under the Clifford inner product? P1.1 and P1.2 test empirically; theoretical analysis would require extending Scellier & Bengio 2017 to non-commutative algebras.

2. **Energy vs. dynamics: which matters more?** Does geometric structure in the energy function provide most of the benefit, with dynamics being secondary? Or does the geometric update rule matter independently? P2.1 + P1.2 together answer this — if BilinearEnergy + LinearDot beats NormEnergy + GeomProduct, energy dominates; if the converse holds, dynamics dominates.

3. **Are Clifford fixed points geometrically interpretable?** When the network settles, does the multivector state have readable geometric content (grade-1 part points toward target class, grade-2 part encodes local orientation), or is it an arbitrary Clifford-format numeric vector? A1 and A2 address this directly — if attractor clusters form symmetry orbits, the answer is yes.

4. **Equivariance through training.** The architecture has equivariance structure at initialization. Do EP weight updates preserve this through training, or does gradient noise break it? A3 tracks this. The SN-off vs. SN-on comparison in A3 directly tests whether spectral normalization is the key to maintaining equivariance during optimization.

5. **The language question.** Is there a form of geometric structure in language — syntactic dependency arcs as oriented edges, compositional structure as geometric product — that Clifford multivectors can exploit? PL1 (character LM) tests basic viability; PL2 SCAN test (compositional generalization) is the decisive test; PL3 (JEPA-style latent prediction) is the most speculative.

6. **CGA vs. Cl(3,0): when is the larger algebra worth it?** CGA Cl(4,1) motors are strictly more expressive for rigid-body tasks (translations + rotations unified) but 4× larger (16D vs. 4D rotor). P1.6 and PM2 answer: does the CGA structure actually help, or does the model learn equivalent representations in Cl(3,0) anyway? This question determines whether CGA is a useful research direction or just an expensive reparametrization.

7. **Scaling to real benchmarks.** Can Clifford-EP match NequIP on QM9 (PM1) and EGNN on N-body (PG1)? These are the two most demanding quantitative tests. If yes, the framework is competitive with the state of the art in equivariant ML on its strongest domains.
