# **Bit-Packed Phase Router**

A **high-performance C++ / Python library** for building **balanced, collision-free bipartite routings** between large sparse binary matrices.

It implements a **deterministic, bit-parallel sampler** for a **degree-capped Chung–Lu bipartite graph** using cyclic phase mixing and permutations.

Designed for:

- Mixture-of-Experts (MoE)
- Sparse attention
- Load-balanced fan-out
- Large bipartite graph coupling
- Stochastic routing at scale

All computation is **bit-packed** (64 bits per word) and uses only:

```
AND, shifts, popcount, and permutations
```

making it memory-bandwidth limited, cache-efficient, and SIMD-friendly.

---

### **CPU-First Design**

The Bit-Packed Phase Router is **designed for CPU execution** to leverage bit-parallel operations, cache-friendly access, and deterministic top-k routing.

For a full discussion of why GPUs are not used, see [Why CPU (and not GPU)?](docs/why_cpu.md).

---

## Build & Run (Quick Start)

### **1. Build C++ backend**

```bash
# From project root
python setup.py build_ext --inplace
```

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

Given two binary matrices:

```
S, T ∈ {0,1}^(N×N)
```

with row sums:

```
s_i = Σ_j S_ij,    t_j = Σ_i T_ij
```

the router constructs up to **k routes per row** by computing:

```
O = S' ∧ (T')^T
```

where `S'` and `T'` are **independently phase-mixed and permuted degree-preserving transforms** of the inputs.

For large `N`:

```
E[O_ij] ≈ s_i * t_j / N
```

Thus the router samples a **Chung–Lu (configuration-model) bipartite graph** with:

- Larger `s_i` → more outgoing routes
- Larger `t_j` → more incoming load

subject to a **hard fan-out cap (k) per source row**.

---

## Guarantees

The output satisfies:

- Expected row sums scale with `s_i`
- Expected column sums scale with `t_j`
- Collisions are uniformly distributed
- No column becomes a geometric or phase-aligned hotspot
- Fan-out per row is bounded by `k`

The degrees are preserved **in expectation**, not exactly per realization.

---

## Why this is useful

In MoE, sparse attention, and distributed routing, we need to map many sources to many targets while:

- Respecting capacity
- Avoiding hotspots
- Avoiding feedback loops
- Keeping runtime cost low

Most routers rely on:

- Hashing
- Learned softmax gates
- Greedy balancing

These are unstable, require training, or need coordination.

The Phase Router instead produces a **random-like bipartite graph by construction**, using only deterministic transforms and bitwise operations.

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
- Preserves column sums while destroying geometric correlations

### **5. Extra Row Permutation for T & Transpose**

- Permute rows of `T` before transposing
- After transpose, optionally apply column permutation
- Ensures statistical independence between `S'` and `T'`

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
O_j = Σ_i O_ij ≈ Poisson(|S| * t_j / N)
```

- Truncated by the **per-row k-cap**
- No hotspots, clusters, or stripes → visually a **starfield**

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

### **Practical Formulas**

```
Row sum:       r_i = Σ_j O_ij  ≤ k
Column sum:    c_j = Σ_i O_ij ≤ t_j
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

## Statistical Analysis & Monte-Carlo Utilities

The router implements a **Monte-Carlo transport operator** that samples random feasible couplings given degree marginals and sparsity constraints. For practical deployment, we provide statistical utilities in `src/router_stats.py`:

### Monte-Carlo Sampling with Theoretical Baseline

```python
from router_stats import monte_carlo_stats

# Get comprehensive statistics with Chung–Lu theoretical baseline
stats = monte_carlo_stats(S, T, k=32, num_samples=50)
print(f"Expected max load: {np.max(stats['mean_load']):.1f}")
print(f"95th percentile: {np.max(stats['p95_load']):.1f}")
print(f"Global skew: {stats['global_skew']:.2f}")
print(f"Theoretical bias: {np.linalg.norm(stats['bias']):.2f}")
print(f"Relative error: {np.mean(np.abs(stats['relative_error'])):.3f}")
```

### Capacity Planning for MoE with Tail Risk

```python
from router_stats import estimate_expert_capacity

# Determine capacity requirements with confidence intervals
capacity = estimate_expert_capacity(S, T, k=32, confidence=0.99)
print(f"99th percentile capacity: {np.max(capacity['required_capacity'])}")
print(f"Mean capacity: {np.mean(capacity['mean_capacity']):.1f}")
print(f"Utilization: {np.mean(capacity['utilization'])*100:.1f}%")
print(f"Headroom needed: {np.mean(capacity['headroom']):.1f}")
```

### Optimal Parameter Search with Binary Search

```python
from router_stats import suggest_k_for_balance

# Find minimal k for desired load balance (10-30x faster with binary search)
result = suggest_k_for_balance(S, T, target_skew=1.5, verbose=True)
print(f"Recommended k: {result['recommended_k']}")
print(f"Achieved skew: {result['achieved_skew']:.2f}")
print(f"Evaluations: {result['num_evaluations']} (binary search efficiency)")
print(f"Success: {'Yes' if result['success'] else 'No'}")
```

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

- **Chung–Lu expectation baseline**: `ideal_mean = k * T.sum(axis=0) / T.sum()`
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

| Metric                | Formula                        | Interpretation                   |
| --------------------- | ------------------------------ | -------------------------------- |
| `mean_load[j]`        | E[L_j]                         | Expected load on column j        |
| `ideal_mean[j]`       | k·t_j/∑t                       | Theoretical Chung–Lu expectation |
| `bias[j]`             | mean_load[j] - ideal_mean[j]   | Deviation from theory            |
| `global_skew`         | max(mean_load)/mean(mean_load) | MoE overload risk                |
| `temporal_skew[j]`    | max(L_j)/mean(L_j)             | Per-column variability           |
| `edge_density`        | ∑mean_load/(N²)                | Empirical sparsity               |
| `theoretical_density` | k/N                            | Expected sparsity                |

## Research Applications

These utilities enable **research-grade analysis** of stochastic routing:

1. **Bias-Variance Tradeoff**: Compare empirical results to Chung–Lu theory
2. **Tail Risk Analysis**: Estimate overflow probabilities for MoE systems
3. **Parameter Optimization**: Find minimal k for desired load balance
4. **Capacity Planning**: Determine expert sizes with confidence intervals
5. **Convergence Studies**: Analyze how routing quality improves with k

All functions maintain the router's core design as a **Monte-Carlo transport operator** while adding rigorous statistical analysis capabilities.

---

## What this is (and is not)

This is:

- A deterministic degree-weighted mixing operator
- A fast Chung–Lu bipartite sampler
- A scalable routing primitive with hard fan-out limits

This is **not**:

- A learned router
- A greedy load balancer
- A hash

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

| Symbol / Expression                         | Meaning                                                                                 |     |     |
| ------------------------------------------- | --------------------------------------------------------------------------------------- | --- | --- |
| `S, T ∈ {0,1}^(N×N)`                        | Binary source and target matrices of size N×N                                           |     |     |
| `s_i = Σ_j S_ij`                            | Row sum of source matrix row i                                                          |     |     |
| `t_j = Σ_i T_ij`                            | Column sum of target matrix column j                                                    |     |     |
| `O = S' ∧ (T')^T`                           | Output routing matrix via bitwise AND of phase-mixed `S'` and transposed `T'`           |     |     |
| `E[O_ij] ≈ s_i * t_j / N`                   | Expected value of each output bit; approximates Chung–Lu configuration-model statistics |     |     |
| `Pr(S'_ik = 1)`                             | Probability that the k-th bit of row i in `S'` is set                                   |     |     |
| `O_j = Σ_i O_ij ≈ Poisson(\|S\| * t_j / N)` | Approximate column load distribution, Poisson with mean proportional to `t_j`           |

| `φ_i^S = Σ_{r<i} s_r` | Phase offset for row i in `S` (cumulative sum for barrel-shifting) | | |
| `fill = (total active routes) / (N * k)` | Fill ratio: fraction of possible routes that are active | | |
| `max(c_j) / mean(c_j)` | Load balance ratio across columns | | |

**Notes:**

- All summations (`Σ`) are over integers and are implemented efficiently with **bit-packed operations** in code.
- Bitwise AND (`∧`) represents the intersection of source and target arcs in the phase-spread matrices.
- Poisson approximation is valid for **large N**, before applying the hard cap `k` per row.
- All formulas are **illustrative**; the actual code performs these operations **in packed 64-bit words** using CPU bitwise instructions.

---
