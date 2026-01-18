# **Bit-Packed Phase Router**

A **high-performance C++ / Python library** for constructing **balanced, randomly mixed bipartite routings** using only **deterministic bitwise operations and permutations**.

The Phase Router implements a **seed-controlled phase-space mixing operator**, also referred to as an **Orthogonal Load-Balanced Incidence Operator (OLBIO)** ([see details](docs/OLBIO.md)), that converts two degree-specified binary matrices into a sparse bipartite coupling which **removes input-order bias and phase alignment effects** under typical conditions.

It is best understood as a **randomized analogy to convolution in cyclic phase space**: row-degree mass from both sides is embedded on a ring, independently mixed by permutations and phase shifts, and intersected to produce a sparse routing whose first-order statistics match the expected-degree law of a Chung–Lu bipartite model.

Unlike Chung–Lu, this is **not an independent-edge sampler**. It is a **constructive, seed-randomized, deterministic transport operator** implemented entirely with bitwise AND, shifts, popcount, and permutations, making it cache-efficient, SIMD-friendly, and fully reproducible.

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
- It is not a hard real-time load balancer
- It does not eliminate Poisson extremes
- It does not guarantee per-column caps
- It is not optimal for k ≪ √N

---

## **What the Phase Router Computes**

Given two binary matrices

```
S, T ∈ {0,1}^(N×N)
```

with row sums

```
s_i = Σ_j S_ij ,   t_j = Σ_j T_ij
```

the Phase Router constructs a sparse bipartite routing matrix `O` with **at most `k` outputs per source row** by embedding both matrices into a **shared cyclic phase space** and intersecting them:

```
O = S' ∧ (T')^T
```

Here `S'` and `T'` are **degree-preserving, seed-randomized, deterministic phase-mixed transforms** of the original inputs.

- Row-degree mass from both sides is packed into contiguous arcs on a ring of size `N`.

- Independent permutations and phase shifts distribute arcs quasi-randomly.

- Bitwise intersection produces a sparse bipartite coupling with first-order statistics:

```
E[O_ij] ≈ s_i · t_j / N
```

This **matches the expected-degree law** of a Chung–Lu bipartite model, but the **joint distribution is fundamentally different**: edges arise from **structured phase overlap**, not independent Bernoulli trials.

The Phase Router is therefore a **low-discrepancy randomized coupling**, not a probabilistic graph sampler.

---

## **Construction / High-Level Algorithm**

The phase-mixed transforms `S'` and `T'` are constructed as follows:

1. **Row Alignment**
   Left-align all 1-bits in each row of `S` and `T` into a contiguous block, preserving row sums.

2. **Phase Offset Computation (Independent for S and T)**
   Compute cumulative row offsets from the **original matrix order**:

   ```
   φ_i^S = Σ_{r<i} s_r  (mod N)
   φ_i^T = Σ_{r<i} t_r  (mod N)
   ```

   These offsets determine how much each row will be cyclically rotated.

3. **Cyclic row-wise bit rotation (Global N-bit Row-wise)**
   Apply independent cumulative cyclic rotations to each row:

   ```
   S_rot[i] = RotateLeft(S[i], φ_i^S)
   T_rot[i] = RotateLeft(T[i], φ_i^T)
   ```

   This embeds each row's contiguous arc on a ring of size `N`, creating **low-discrepancy phase spreading** that reduces collisions.

4. **Column Permutations**
   Apply independent random column permutations to `S_rot` and `T_rot`:

   ```
   S_shuf = PermuteColumns(S_rot, col_perm_S)
   T_shuf = PermuteColumns(T_rot, col_perm_T)
   ```

   Column permutations redistribute bits horizontally without altering the row-wise phase spreading effect. This preserves row sums while destroying geometric correlations. Each matrix uses a different seed-derived permutation.

5. **OPTIONAL: Global Row Permutations (Independent for S and T)**
   Apply independent row permutations to both matrices:

   ```
   S_final[i] = S_shuf[row_perm[i]]
   T_prepared[i] = T_shuf[row_perm_T[i]]
   ```

   This removes input-order bias. Note that `S` and `T` use **different** row permutations to ensure independent mixing.

   The **row permutation step** is **not part of the core algorithm**. Its main purposes are:

   - **Anonymization / obfuscation** of input order
   - Minor reduction of skew for very small k values

   **Trade-offs:**

   - **Computational cost:** Each row permutation adds memory accesses and bookkeeping overhead, which can noticeably increase runtime for large N.
   - **Marginal benefit:** In two-phase adversarial or large-scale scenarios, row permutations have **little to no impact** on column skew or maximum loads.

   **Usage:**

   - Row permutations can be toggled via the `ENABLE_ROW_PERM` compile-time flag (`#ifdef ENABLE_ROW_PERM`) or the corresponding runtime API argument.
   - By default, they are **disabled** to maximize performance.
   - Recommended only for testing, anonymization, or seed-independent input-order mixing.

   **Performance Note:**

   Empirical tests show that disabling row permutations:

   - Reduces runtime by up to **30–50%** for large N (≥1024)
   - Has **negligible effect on skew** for k ≥ 256
   - Preserves the Phase Router’s **deterministic, low-discrepancy behavior**

6. **Transpose T 90° Clockwise**
   Rotate T 90° clockwise (transpose) to align its columns with the output rows:

   ```
   T_final[j][i] = T_prepared[i][j]
   ```

   This preserves the phase offsets applied in Step 3 while preparing `T` for the intersection.

7. **Bitwise Intersection & Seed-deterministic Randomized Top-k Extraction**

   ```
   O[i] = S_final[i] & T_final[i]
   ```

   Only up to k intersections per row are retained. By default, these are chosen via seed-deterministic random selection among candidate overlaps, rather than always taking the top-k. This reduces column skew and distributes load more evenly, especially for small k, while hard fan-out bound is enforced. If fewer than k overlaps exist, the remaining slots are filled with -1.

---

### **Guarantees**

- **Hard Fan-Out Bound**:

  ```
  Σ_j O_ij ≤ k
  ```

  Deterministically enforced for every source row.

- **Degree-Weighted Load**:

  Seed-averaged column load satisfies

  ```
  E[c_j] ∝ t_j ,   where c_j = Σ_i O_ij
  ```

- **No Geometric Bias**:
  Columns cannot become hotspots due to input ordering, contiguity, or phase alignment.

- **Deterministic Reproducibility**:

  Fixed seeds produce identical routings.

---

### **Interpretation & Practical Relevance**

The Phase Router is:

- A **degree-weighted bipartite transport operator**

- **Seed-randomized but deterministic**

- **CPU-efficient and bit-parallel**

It **does not** solve a global optimization problem or sample edges independently.

Practical use cases include Mixture-of-Experts (MoE), sparse attention, and load-balanced routing:

- Enforces capacity constraints

- Avoids hotspots

- Prevents feedback loops

- Requires no training or coordination

- Provides predictable first-order statistical behavior

---

## **Comparison with Alternative Routing Approaches**

| Router Type      | Learned | Deterministic | Load Guarantees | Fan-Out Bound | Global State |
| ---------------- | ------- | ------------- | --------------- | ------------- | ------------ |
| Softmax MoE      | ✓       | ✗             | Weak            | ✗             | ✓            |
| Hash Router      | ✗       | ✓             | Poor            | ✓             | ✗            |
| Greedy Balancer  | ✗       | ✓             | Strong          | ✓             | ✓            |
| **Phase Router** | ✗       | ✓             | **Statistical** | **✓**         | ✗            |

**Hash Router vs Phase Router**

| Feature                    | Hash Router        | Phase Router         |
| -------------------------- | ------------------ | -------------------- |
| Column Skew                | Higher (worse)     | Lower (better)       |
| Max Column Load            | Higher / bursty    | Lower / balanced     |
| Runtime                    | Faster             | Slower               |
| Determinism                | Seed-deterministic | Seed-deterministic   |
| Handling Structured Inputs | Prone to hotspots  | Smooth, decorrelated |

For more detail, see [`phase_router_vs_hash_router_tables.md`](docs/phase_router_vs_hash_router_tables.md)

**Chung–Lu vs Phase Router**:

| Feature         | Chung–Lu          | Phase Router             |
| --------------- | ----------------- | ------------------------ |
| Randomness      | Independent edges | Structured phase overlap |
| Edge Caps       | None              | Hard k-cap               |
| Reproducibility | ✗                 | ✓ (seed-deterministic)   |
| Geometry Bias   | ✗                 | Explicit phase mixing    |

The Phase Router produces lower column skew and more balanced routing compared with a hash router, while remaining deterministic and enforcing a hard per-row fan-out limit. It is well-suited for large-scale MoE routing, sparse attention, or other bipartite routing tasks where predictable load distribution and handling of structured inputs are important, even if it is somewhat slower than a simple hash-based approach.

---

## **CPU-First Design & Performance**

### **Performance Highlights**

The Phase Router achieves **exceptional performance** through bit-packed operations and cache-optimized permutations:

- **Sub-millisecond routing** for N ≤ 512 (typical MoE token routing scale)
- **~16 ms** for N=1024, k=256 (production MoE configurations)
- **~78 ms** for N=4096, k=512 (large-scale sparse routing)
- **Linear scaling** with matrix size for fixed k
- **Memory-bandwidth limited**, not compute-limited

### **Benchmarks (Intel Core i5-2410M @ 2.30 GHz, 8Gb RAM, 2 cores)**

| N    | k   | Routing Time | Total Time |
| ---- | --- | ------------ | ---------- |
| 256  | 64  | 0.55 ms      | 0.68 ms    |
| 512  | 256 | 2.35 ms      | 2.84 ms    |
| 1024 | 256 | 15.61 ms     | 17.58 ms   |
| 2048 | 512 | 29.71 ms     | 42.15 ms   |
| 4096 | 512 | 77.95 ms     | 110.76 ms  |

> **Note:** PBM dumping disabled; PBM output adds (O(N^2)) memory and file I/O overhead that dominates runtime at large (N).

**Key observations:**

- Disabling debug dumps yields **30-157× speedup** (dumping incurs PBM I/O overhead)
- Routing time dominates for large N (memory bandwidth)
- Packing overhead visible only for small N (< 1 ms)
- Suitable for amortized or batch-level routing in real-time MoE and sparse attention systems
- **No GPU required**: optimized for cache-coherent SIMD hardware
  See [`why_cpu.md`](docs/why_cpu.md)

### **Architectural Notes**

- Bit-parallel operations using 64-bit words + popcount/ctz intrinsics
- OpenMP parallelization for row-level operations
- Permutations are cache-friendly (sequential reads after shuffle)
- Phase rotations wrap across word boundaries with minimal branching

---

## **Build & Run (Quick Start)**

### **Dependencies**

- Required: Python 3.x, NumPy, pybind11, setuptools, C++ compiler (`g++`, `clang++`, MSVC)
- Optional: OpenMP (multi-threading), pandas, matplotlib, pillow, SciPy, PyTorch

Install dependencies:

```bash
# From project root
pip install -r requirements.txt
```

> Ensure your virtual environment is activated before installing.

## Quick Start

### **1. Build C++ Backend**

```bash
python setup.py build_ext --inplace
```

### **2. Run Evaluation / Scaling Experiments**

- **Single test** (N = 256, k = 32):

```bash
python evaluation/phase_router_test.py
```

- **Batch execution script** for comprehensive scaling experiments

```bash
python evaluation/phase_router_run.py
```

- **Stress test** comparing the bit-packed **phase router** against a simple **hash-based router**

```bash
python evaluation/phase_router_vs_hash.py --skip-plots
```

(Plots optional)

[Stress test documentation and usage instructions](docs/phase_router_vs_hash.md)

- **10-point multi-test** to probe **load balance, determinism, composability, and failure modes**

```bash
python evaluation/phase_router_test_matrix.py
```

[Test matrix documentation and usage instructions](docs/phase_router_test_matrix.md)

### **3. Examples**

- **Demo with random binary matrices** (NumPy/PyTorch, routes + stats output):

```bash
python examples/demo_router.py
```

- **MoE capacity planning** and overflow analysis:

```bash
python examples/moe_routing_demo.py
```

---

## **Python API**

```python
stats = router.pack_and_route(S, T, k, routes, seed=42)
```

- Runs `align → phase → permutations → transpose → AND → top-k extraction`

- Deterministic given a seed

**Advanced API:** Users can provide their own packed arrays and permutations:

```python
route_packed_with_stats(...)
```

---

## **Statistical Analysis & Seed-Ensemble Evaluation**

- Each seed produces a fully deterministic routing:

```
fixed `(S, T, k)` + seed → deterministic routing `O(seed)`
```

- Averaging over seeds approximates expected-degree behavior.
- Statistical quantities are **seed-averaged**, not independent-edge estimates

### Utilities in `src/router_stats.py`:

- `monte_carlo_stats()` → seed-averaged loads, biases, skew

- `estimate_expert_capacity()` → percentile-based capacity planning

- `suggest_k_for_balance()` → parameter search over k

- `analyze_routing_distribution()` → theoretical vs empirical densities

**Key Features:**

- Vectorized column load computation

- Binary search for optimal k

- Tail-risk estimation for MoE

- Deterministic reproducibility

---

## **Statistical Metrics Reference**

| Metric                | Formula                        | Interpretation                         |
| --------------------- | ------------------------------ | -------------------------------------- |
| `mean_load[j]`        | E[L_j]                         | Expected load on column j              |
| `ideal_mean[j]`       | k·t_j/∑t                       | Seed-averaged expected-degree baseline |
| `bias[j]`             | mean_load[j] - ideal_mean[j]   | Deviation from theory                  |
| `global_skew`         | max(mean_load)/mean(mean_load) | MoE overload risk                      |
| `temporal_skew[j]`    | max(L_j)/mean(L_j)             | Per-column variability                 |
| `edge_density`        | ∑mean_load/(N²)                | Empirical sparsity                     |
| `theoretical_density` | k/N                            | Expected sparsity                      |

## **Note:** These are seed-averaged and not Monte-Carlo independent-edge estimates

## **Adversarial Inputs**

- This is not a cryptographic primitive

- Guarantees apply to benign or random-like degree inputs

- Adversarially constructed inputs may violate mixing assumptions if seed is known

---

## **Experimental Setup**

- Matrix sizes N: 256 → 4096

- Max connections per row/column k: 8 → 512

- Trials: 3 runs per config; reproducibility tests: 5 runs

- Metrics collected: row/column statistics, fill ratio, runtime

- Routing: `O = S' ∧ T'^T` using fully bit-packed implementation

---

## **Equation Notes**

| Symbol / Expression                      | Meaning                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| `S, T ∈ {0,1}^(N×N)`                     | Binary source and target matrices of size N×N                          |
| `s_i = Σ_j S_ij`                         | Row sum of source row i                                                |
| `t_j = Σ_i T_ij`                         | Column sum of target column j                                          |
| `O = S' ∧ (T')^T`                        | Output routing via bitwise AND of phase-mixed `S'` and transposed `T'` |
| `E[O_ij] ≈ s_i * t_j / N`                | Expected output bit; first-order Chung–Lu degree approximation         |
| `Pr(S'_ik = 1)`                          | Probability that the k-th bit of row i in `S'` is set                  |
| `O_j = Σ_i O_ij`                         | Column load induced by structured phase overlap                        |
| `φ_i^S = Σ_{r<i} s_r`                    | Phase offset for row i in `S`                                          |
| `fill = (total active routes) / (N * k)` | Fill ratio                                                             |
| `max(c_j) / mean(c_j)`                   | Load balance ratio across columns                                      |

---

The theoretical construction is documented in [`theory.md`](docs/theory.md)

The testing suite is documented in [`Testing Suite.md`](docs/Testing_Suite.md)

Empirical performance and load-balance results are documented in [`evaluation.md`](docs/evaluation.md)

Potential applications are suggested in [`applications.md`](docs/applications.md)

---
