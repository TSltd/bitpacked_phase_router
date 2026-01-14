# **Bit-Packed Phase Router**

A **high-performance C++ / Python library** for constructing **balanced, randomly mixed bipartite routings** using only **deterministic bitwise operations and permutations**.

The Phase Router implements a **seed-controlled phase-space mixing operator** that converts two degree-specified binary matrices into a sparse bipartite coupling with **no geometric or phase-aligned bias**.

It is best understood as a **randomized convolution in cyclic phase space**: row-degree mass from both sides is embedded on a ring, independently mixed by permutations and phase shifts, and intersected to produce a sparse routing whose first-order statistics match the expected-degree law of a Chung–Lu bipartite model.

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

2. **Global Row Permutation**
   Apply a shared random permutation to both matrices to remove input-order bias.

3. **Phase Spreading (Independent for S and T)**
   For permuted row `i`:

   ```
   φ_i^S = Σ_{r<i} s_r
   φ_i^T = Σ_{r<i} t_r
   ```

   Cyclically rotate each row by its offset, embedding a contiguous arc on a ring of size `N`. This creates **low-discrepancy phase spreading**, reducing collisions.

4. **Column Permutations**
   Apply independent random column permutations to `S` and `T`, preserving row sums while destroying geometric correlations.

5. **Extra Row Permutation & Transpose for T**
   Permute rows of `T` before transposing to ensure `S'` and `(T')^T` are independently mixed degree-preserving fields.

6. **Bitwise Intersection & Top-k Extraction**

   ```
   O_ij = Σ_ℓ S'_iℓ ∧ T'_jℓ
   ```

   Bit-packed AND and popcount compute overlaps efficiently. Only the first `k` intersections per row are retained, enforcing a **hard fan-out bound**.

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

**Chung–Lu vs Phase Router**:

| Feature         | Chung–Lu          | Phase Router             |
| --------------- | ----------------- | ------------------------ |
| Randomness      | Independent edges | Structured phase overlap |
| Edge Caps       | None              | Hard k-cap               |
| Reproducibility | ✗                 | ✓ (seed-deterministic)   |
| Geometry Bias   | ✗                 | Explicit phase mixing    |

---

## **CPU-First Design & Performance**

- Memory-bandwidth limited, permutation heavy, branchy, bit-parallel
- Optimized for cache-coherent SIMD hardware; GPUs often underperform
- Example N=4096:

| Stage                | Time    |
| -------------------- | ------- |
| Phase + Permutations | ~176 ms |
| Bit-packed AND       | ~327 ms |
| Total Routing        | ~0.5 s  |

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

### **1. Build C++ Backend**

```bash
python setup.py build_ext --inplace
```

### **2. Run Evaluation / Scaling Experiments**

- Full scaling test (N = 256–4096, k = 8–512):

```bash
python evaluation/phase_router_run.py
```

### **3. Quick Test**

- Single test (N = 256, k = 32):

```bash
python evaluation/phase_router_test.py
```

### **4. Examples**

- Run demo with random binary matrices (NumPy/PyTorch, routes + stats output):

```bash
python examples/demo_router.py
```

- MoE capacity planning and overflow analysis:

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

- Each run: fixed `(S, T, k)` + seed → deterministic routing `O(seed)`

- Statistical quantities are **seed-averaged**, not independent-edge estimates

Utilities in `src/router_stats.py`:

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

---
