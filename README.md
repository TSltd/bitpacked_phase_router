# **Bit-Packed Phase Router**

A **high-performance C++ / Python library** for constructing **balanced, randomly mixed bipartite routings** using only **deterministic bitwise operations and permutations**.

The Phase Router implements a **seed-controlled phase-space mixing operator** that converts two degree-specified binary matrices into a sparse bipartite coupling with **no geometric or phase-aligned bias**.

It is best understood as a **randomized convolution in cyclic phase space**: row-degree mass from two sides is embedded on a ring, independently mixed by permutations and phase shifts, and intersected to produce a sparse routing whose first-order statistics match the expected-degree law of a Chung–Lu bipartite model.

Unlike Chung–Lu, this is not an independent-edge sampler.
It is a **constructive seed-randomized deterministic transport operator** implemented entirely with:

```
bitwise AND, shifts, popcount, and permutations
```

making it cache-efficient, SIMD-friendly, and fully reproducible.

---

## What this library does

Given two binary matrices

```
S, T ∈ {0,1}^(N×N)
```

with row sums

```
s_i = Σ_j S_ij ,   t_i = Σ_j T_ij
```

the router produces a sparse bipartite routing matrix `O` with at most `k` outputs per row by embedding both matrices into a **shared cyclic phase space** and intersecting them:

```
O = S' ∧ (T')^T
```

Here `S'` and `T'` are **degree-preserving, randomly phase-mixed transforms** of the original matrices.

The transformation consists of:

• left-aligning all 1-bits in each row
• randomly permuting rows
• assigning each row a cumulative phase offset
• applying independent column permutations
• permuting and transposing `T`
• intersecting the two packed fields
• keeping only the first `k` matches per row

All operations preserve row sums and use only deterministic bit-parallel primitives.

---

## Statistical meaning

After phase mixing, each row becomes a uniformly random arc on a ring of size `N`.

For large `N`, this implies

```
Pr[S'_ik = 1] ≈ s_i / N
Pr[T'_jk = 1] ≈ t_j / N
```

so each output entry behaves like a randomized convolution:

```
E[O_ij] ≈ s_i · t_j / N
```

This matches the expected-degree law of a Chung–Lu bipartite model, but the joint distribution is completely different: edges arise from structured phase overlap, not independent Bernoulli trials — they arise from structured overlap of two phase-mixed fields.

The router therefore implements a **low-discrepancy randomized coupling**, not a probabilistic graph sampler.

---

## Guarantees

The Phase Router provides the following guarantees:

### 1. Hard fan-out bound

Each source row produces at most `k` outputs:

```
Σ_j O_ij ≤ k
```

This is enforced deterministically.

---

### 2. Degree-weighted load

Seed-averaged column load is proportional to target weights:

```
E[c_j] ∝ t_j ,   where  c_j = Σ_i O_ij
```

Targets with higher demand receive more routes on average.

---

### 3. No geometric bias

Because all arcs are phase-shifted and permuted, no column can become a hotspot due to:

• input ordering
• contiguous structure
• phase alignment
• adversarial layout

Any load variation is statistical, not algorithmic.

---

### 4. Randomized mixing

The output distribution depends only on:

• the degree sequences `{s_i}`, `{t_j}`
• the random seed

not on matrix geometry.

---

### 5. Deterministic reproducibility

For a fixed seed, identical inputs produce identical routings.
This supports reproducible experiments and Monte-Carlo evaluation over random phase embeddings.

---

## Why this is useful

In MoE, sparse attention, and large-scale routing, we must map many sources to many targets while:

• respecting capacity
• avoiding hotspots
• preventing feedback loops
• staying fast

Most systems rely on learned gates, hashing, or greedy balancing.
The Phase Router instead produces a **random-like bipartite coupling by construction**, using only deterministic transforms.

It gives you:

• degree-weighted load
• bounded fan-out
• no coordination
• no training
• no geometry leaks

Unlike flow-based or b-matching approaches, the Phase Router does not solve a global optimization problem; it implements a fast randomized transport with predictable first-order statistical behavior.

---

## Comparison with Alternative Routing Constructions

| Router type      | Learned | Deterministic | Load guarantees | Fan-out bound | Global state |
| ---------------- | ------- | ------------- | --------------- | ------------- | ------------ |
| Softmax MoE      | ✓       | ✗             | Weak            | ✗             | ✓            |
| Hash router      | ✗       | ✓             | Poor            | ✓             | ✗            |
| Greedy balancer  | ✗       | ✓             | Strong          | ✓             | ✓            |
| **Phase Router** | ✗       | ✓             | **Statistical** | **✓**         | ✗            |

## Build & Run (Quick Start)

### **1. Build C++ backend**

```bash
# From project root
python setup.py build_ext --inplace
```

---

## CPU-first design

The Phase Router is optimized for CPUs because it is:

• memory-bandwidth limited
• permutation heavy
• branchy
• bit-parallel

It maps naturally to cache-coherent SIMD hardware.
GPUs are not required and often underperform for this workload.

---

### **2. Run evaluation / scaling experiments**

```bash
# From project root
python evaluation/phase_router_run.py
```

- Results saved under `evaluation_results/`:

  - `N*_k*_trial*/` → individual run outputs (JSON + PBM/PNG)
  - `figures/` → performance plots (PNG)
  - `reproducibility/` → reproducibility test results
  - `summary.csv` → aggregated metrics

### **3. Quick test (optional)**

```bash
python tests/phase_router_test.py
```

- Runs a small routing test with validation and prints metrics to console.

---

**Notes:**

- Requires Python 3.x, NumPy, and a C++ compiler (`g++`, `clang++`, or MSVC).
- Optional: OpenMP for multi-threaded execution.
- Scripts are CPU-first and deterministic with fixed seeds.

---

## What it computes

Given two binary matrices

```
S, T ∈ {0,1}^(N×N)
```

with row sums

```
s_i = Σ_j S_ij ,   t_i = Σ_j T_ij
```

the router constructs up to **k routes per row** by embedding both matrices into a **shared cyclic phase space** and intersecting them:

```
O = S' ∧ (T')^T
```

where `S'` and `T'` are **degree-preserving, seed-randomized, deterministic phase-mixed transforms** of the original inputs.

The construction works as follows:

- Each row is converted into a **contiguous arc** of length `s_i` (for `S`) or `t_i` (for `T`) on a ring of size `N`
- The arcs are **randomly permuted and phase-shifted** using cumulative sums and permutations
- The two embedded fields are then **intersected bitwise**
- For each row, only the first `k` intersections are kept

This produces a sparse bipartite coupling whose statistics depend only on the row-sum sequences `{s_i}` and `{t_j}` and on the random seed.

---

## Statistical meaning

After phase mixing, each row occupies a uniformly random arc on the ring.
For large `N`, this implies

```
Pr[S'_iℓ = 1] ≈ s_i / N
Pr[T'_jℓ = 1] ≈ t_j / N
```

so the intersection behaves like a randomized convolution:

```
E[O_ij] ≈ s_i · t_j / N
```

This matches the **first-moment structure** of a Chung–Lu bipartite model, but **the edges are not independent** — they are produced by structured phase interactions.

The router therefore implements a **low-discrepancy randomized coupling**, not an independent-edge sampler.

---

## High-Level Algorithm & Statistical Behavior

### **1. Align Rows**

- Left-align all 1s in each row of `S` and `T`
- Preserves **row sums**

```
s_i = Σ_j S_ij,    t_j = Σ_i T_ij
```

### **2. Global Row Permutation**

- Apply the same random row permutation to both `S` and `T`
- Randomizes phase placement, removing input order bias

### **3. Phase Spreading (Independent for S and T)**

- For permuted row i:

```
φ_i^S = Σ_{r<i} s_r
φ_i^T = Σ_{r<i} t_r
```

- Cyclically rotate each row by its offset
- Embeds contiguous "arcs" on a ring of size `N`
- Acts like **low-discrepancy phase spreading** → reduces collisions

### **4. Column Permutations**

- Apply **independent column permutations** to `S` and `T`
- Preserve row histograms embedded into a ring

### **5. Extra Row Permutation for T & Transpose**

- Permute rows of `T` before transposing
- After transpose, optionally apply column permutation
- Ensures two correlated random tilings of a ring

### **6. Bitwise AND & Top-k Extraction**

```
O_ij = Σ_k S'_ik ∧ T'_jk
```

- Use **bit-packed AND** and `popcount`
- For each source row, emit **first k hits**; discard extra
- Enforces **hard fan-out cap** per row

---

### **Statistical Behavior**

After phase mixing:

```
Pr(S'_ik = 1) ≈ s_i / N,    Pr(T'_jk = 1) ≈ t_j / N
```

Expected output:

```
E[O_ij] ≈ s_i * t_j / N
```

Column loads:

```
E[Oj​]=i∑​E[Oij​]=i∑​Nsi​tj​​=Ntj​​i∑​si​=N∣S∣tj​​
```

• Asymptotically degree-weighted
• Weakly correlated across rows
• More regular (lower variance) than independent Chung–Lu sampling
• Truncated by the per-row k-cap

---

### **Algorithm Summary Table**

| Step | Operation                       | Purpose                             |
| ---- | ------------------------------- | ----------------------------------- |
| 1    | Align rows                      | Preserve row sums                   |
| 2    | Global row permutation          | Remove input bias                   |
| 3    | Phase spreading                 | Uniform arc distribution            |
| 4    | Column permutation              | Preserve sums, destroy correlations |
| 5    | Row permutation + Transpose (T) | S/T independence                    |
| 6    | Bitwise AND + Top-k             | Hard fan-out routing                |

---

### **Comparison with Chung–Lu**

Chung–Lu is a **probability model**.
The phase router is a **constructive transport operator**.

It does not sample edges — it **flows mass through phase space**.

That gives:

| Chung–Lu           | Phase Router           |
| ------------------ | ---------------------- |
| Random graph       | Randomized transport   |
| Independent edges  | Structured convolution |
| No caps            | Hard k-cap             |
| No reproducibility | Seed-deterministic     |
| No geometry        | Explicit phase mixing  |

---

### **Practical Formulas**

```
Row sum:       r_i = Σ_j O_ij  ≤ k
Column sum:    E[cj​]∝tj​
Fill ratio:    fill = (total active routes) / (N * k)
Load balance:  max(c_j) / mean(c_j)
```

---

## Performance

For N=4096:

| Stage                | Time    |
| -------------------- | ------- |
| Phase + permutations | ~176 ms |
| Bit-packed AND       | ~327 ms |
| Total routing        | ~0.5 s  |

The kernel is **memory-bandwidth limited**, not compute-limited.

---

## Python API

```python
stats = router.pack_and_route(S, T, k, routes)
```

Runs:

```
align → phase → permutations → transpose → AND → top-k extraction
```

### Reproducible routing

```python
stats = router.pack_and_route(S, T, k, routes, seed=42)
```

With a fixed seed, identical inputs produce identical routings.

---

### Advanced API

Advanced users can provide their own packed arrays and permutations via:

```python
route_packed_with_stats(...)
```

---

## Statistical Analysis & Seed-Ensemble Evaluation

The Phase Router is a seed-randomized deterministic transport operator.
All statistical analysis is therefore performed over an ensemble of random phase embeddings, not over independently sampled graphs.

Each run corresponds to:

- fixed inputs (S, T, k)

- a fixed seed

- a deterministic routing outcome O(seed)

Statistical quantities are defined as seed-averaged properties of this operator.

For practical deployment, we provide statistical utilities in `src/router_stats.py`:

### Seed-Ensemble Statistics with Expected-Degree Baseline

```python
from router_stats import monte_carlo_stats

# Evaluate statistics over random phase embeddings
stats = monte_carlo_stats(S, T, k=32, num_samples=50)

print(f"Seed-averaged max load: {np.max(stats['mean_load']):.1f}")
print(f"95th percentile over seeds: {np.max(stats['p95_load']):.1f}")
print(f"Global skew (mean): {stats['global_skew']:.2f}")
print(f"Bias vs expected-degree baseline: {np.linalg.norm(stats['bias']):.2f}")
print(f"Mean relative error: {np.mean(np.abs(stats['relative_error'])):.3f}")

```

Here:

- mean_load[j] is the average column load across seeds

- p95_load[j] estimates tail risk over random embeddings

- bias[j] measures deviation from the expected-degree baseline
  k · t_j / ∑ t

No assumption of independent edges is made.

### Capacity Planning via Seed-Level Tail Risk

```python
from router_stats import estimate_expert_capacity

capacity = estimate_expert_capacity(S, T, k=32, confidence=0.99)

print(f"99th percentile required capacity: {np.max(capacity['required_capacity'])}")
print(f"Mean utilization: {np.mean(capacity['utilization']) * 100:.1f}%")
print(f"Average headroom: {np.mean(capacity['headroom']):.1f}")

```

### Parameter Search over Seed-Averaged Behavior

```python
from router_stats import suggest_k_for_balance

result = suggest_k_for_balance(S, T, target_skew=1.5, verbose=True)

print(f"Recommended k: {result['recommended_k']}")
print(f"Achieved seed-averaged skew: {result['achieved_skew']:.2f}")
print(f"Number of evaluations: {result['num_evaluations']}")
```

This finds the smallest k such that load balance holds uniformly across random phase embeddings, not just in expectation.

### Interpretation

- Randomness enters only through phase embeddings

- Outputs are deterministic given a seed

- Statistics measure robustness to embedding randomness

- Variance reflects transport regularity, not sampling noise

This is fundamentally different from Monte-Carlo graph sampling and more appropriate for systems deployment, where reproducibility and worst-case behavior matter.

### Comprehensive Analysis with Theoretical Density

```python
from router_stats import analyze_routing_distribution

# Analyze behavior across multiple k values
analysis = analyze_routing_distribution(S, T)
for result in analysis['analysis_results']:
    print(f"k={result['k']}: skew={result['global_skew']:.2f}, "
          f"edge_density={result['edge_density']:.3f}, "
          f"theoretical={result['theoretical_density']:.3f}")
```

## Key Features

- **Vectorized column load computation** using `np.bincount` instead of nested loops
- **Binary search** in `suggest_k_for_balance` (O(log n) instead of O(n))

### **Theoretical Rigor**

- **Expected-degree baseline**: `| ideal_mean[j] | k·t_j/∑t | Expected-degree load baseline |`
- **Bias and error metrics**: Compare empirical results to theoretical predictions
- **Proper statistical naming**: `temporal_skew` vs `global_skew`, `edge_density` vs `theoretical_density`

### **Mathematical Correctness**

- **Fixed misleading assignments**: `router.pack_and_route()` returns void, not stats
- **Proper density calculations**: `edge_density = total_routes / (N*N)` and `theoretical_density = k/N`
- **Clear statistical interpretations**: All metrics have precise mathematical meanings

### **Production-Ready Features**

- **Tail risk estimation**: Percentile-based capacity planning for MoE systems
- **Deterministic reproducibility**: All functions support fixed seeds
- **Efficient parameter search**: Binary search finds optimal k in logarithmic time

## Statistical Metrics Reference

| Metric                | Formula                        | Interpretation                           |
| --------------------- | ------------------------------ | ---------------------------------------- |
| `mean_load[j]`        | E[L_j]                         | Expected load on column j                |
| `ideal_mean[j]`       | k·t_j/∑t                       | Expected-degree baseline (seed-averaged) |
| `bias[j]`             | mean_load[j] - ideal_mean[j]   | Deviation from theory                    |
| `global_skew`         | max(mean_load)/mean(mean_load) | MoE overload risk                        |
| `temporal_skew[j]`    | max(L_j)/mean(L_j)             | Per-column variability                   |
| `edge_density`        | ∑mean_load/(N²)                | Empirical sparsity                       |
| `theoretical_density` | k/N                            | Expected sparsity                        |

## Research Applications

These utilities enable **research-grade analysis** of stochastic routing:

1. **Bias-Variance Tradeoff**: Compare empirical results to Chung–Lu theory
2. **Tail Risk Analysis**: Estimate overflow probabilities for MoE systems
3. **Parameter Optimization**: Find minimal k for desired load balance
4. **Capacity Planning**: Determine expert sizes with confidence intervals
5. **Convergence Studies**: Analyze how routing quality improves with k

All functions maintain the router's core design as a **seed-ensemble transport operator** while adding rigorous statistical analysis capabilities.

---

## What this is (and is not)

This is:

- A deterministic degree-weighted mixing operator
- A fast expected-degree bipartite transport operator
- A scalable routing primitive with hard fan-out limits

This is **not**:

- A learned router
- A greedy load balancer
- A hash

---

## Adversarial Inputs

This is not a cryptographic primitive. The Phase Router assumes benign or random-like degree inputs and produces deterministic, reproducible routings for a fixed seed. If the seed is known, an adversary could construct inputs S and T that correlate with the internal permutations, affecting statistical mixing. The router's guarantees apply only in non-adversarial or stochastic settings.

---

## Experimental Setup

- **Hardware**: Intel Core i5-2410M CPU @ 2.30 GHz, 2 cores / 4 threads, 8 GB RAM
- **Compiler / Build**: Python `setup.py build_ext --inplace` (uses system C++ compiler, e.g., `g++`)
- **Software**: Python 3.x, NumPy, optional PyTorch; OpenMP enabled for multi-threaded runs
- **Matrix sizes (N)**: 256 → 4096
- **Maximum connections per row/column (k)**: 8 → 512
- **Number of trials**: 3 independent runs per configuration; reproducibility tests run 5 repeated runs with fixed seeds
- **Inputs**: Random binary matrices `S` and `T` with prescribed row and column sums
- **Routing**: `O = S' ∧ T'^T` in fully bit-packed implementation
- **Metrics collected**: Active routes, routes per row, fill ratio, column statistics (min, max, mean, std, skew), runtime (packing, routing, total), optional PBM/PNG visual outputs

All reported times are **mean ± standard deviation** across trials. Detailed testing procedures, statistical analyses, visual evaluations, and reproducibility protocols are described in [`phase_router_test.py`](docs/Testing_Suite.md) and [`phase_router_run.py`](docs/Testing_Suite.md).

---

The theoretical construction is documented in [`theory.md`](docs/theory.md)
The testing suite is documented in [`Testing Suite.md`](docs/Testing_Suite.md)
Empirical performance and load-balance results are documented in [`evaluation.md`](docs/evaluation.md)

---

## Equation Notes

Some mathematical expressions are written in **plain-text Markdown-friendly form**. Here’s a quick reference:

| Symbol / Expression                      | Meaning                                                                                                        |     |     |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --- | --- |
| `S, T ∈ {0,1}^(N×N)`                     | Binary source and target matrices of size N×N                                                                  |     |     |
| `s_i = Σ_j S_ij`                         | Row sum of source matrix row i                                                                                 |     |     |
| `t_j = Σ_i T_ij`                         | Column sum of target matrix column j                                                                           |     |     |
| `O = S' ∧ (T')^T`                        | Output routing matrix via bitwise AND of phase-mixed `S'` and transposed `T'`                                  |     |     |
| `E[O_ij] ≈ s_i * t_j / N`                | Expected value of each output bit; approximates the expected-degree law of Chung–Lu, not its edge distribution |     |     |
| `Pr(S'_ik = 1)`                          | Probability that the k-th bit of row i in `S'` is set                                                          |     |     |
| `O_j = Σ_i O_ij`                         | Column load induced by structured phase overlap (not an independent-edge model)                                |
| `φ_i^S = Σ_{r<i} s_r`                    | Phase offset for row i in `S` (cumulative sum for barrel-shifting)                                             |     |     |
| `fill = (total active routes) / (N * k)` | Fill ratio: fraction of possible routes that are active                                                        |     |     |
| `max(c_j) / mean(c_j)`                   | Load balance ratio across columns                                                                              |     |     |

**Notes:**

- All summations (`Σ`) are over integers and are implemented efficiently with **bit-packed operations** in code.
- Bitwise AND (`∧`) represents the intersection of source and target arcs in the phase-spread matrices.
- For large N and moderate k, column loads concentrate around their expected-degree baseline, with lower variance than independent Chung–Lu sampling due to negative correlations induced by phase packing and hard caps.
- All formulas are **illustrative**; the actual code performs these operations **in packed 64-bit words** using CPU bitwise instructions.

---
