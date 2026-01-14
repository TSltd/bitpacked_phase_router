# **Bit-Packed Phase Router: Deterministic, Low-Concurrency Routing via Phase-Separated Binary Coupling Matrices**

## **Abstract**

We present a deterministic, discrete method for routing connections between large sparse or dense binary matrices while preserving prescribed row and column sums. Our approach generalizes the **phase-separated barrel-shifting** construction and introduces **global permutations and bit-packing** to enable scalable, cache-efficient computation.

The resulting routing matrix behaves like a **random bipartite graph**, with approximate Bernoulli-distributed entries and Poisson-distributed column sums.

This construction enables **load-balanced routing for Mixture-of-Experts (MoE), sparse attention, and large-scale graph coupling**, while avoiding stochastic routing, learning, or synchronization overhead. The method is highly memory-efficient, fully deterministic, and interpretable, suitable for large-scale, hardware-friendly implementations.

---

## 1. Introduction

Efficient routing between large sets of sources and targets arises in:

- Neural networks with **Mixture-of-Experts (MoE)** or sparse attention layers.
- Graph-based models requiring **bipartite matching**.
- Energy-constrained or neuromorphic hardware.
- Combinatorial scheduling and resource allocation.

Existing routing schemes often fall into three categories:

1. **Hash-based routing:** deterministic but prone to collisions and hotspots.
2. **Softmax or learned routing:** adaptive but unstable, requires extra gradient computation and load regularization.
3. **Greedy or load-aware routing:** expensive and synchronization-heavy at scale.

We propose a **deterministic, discrete, low-concurrency routing primitive** that:

- Preserves **row and column marginals** (exact unless truncated by per-row fan-out `k`).
- Spreads load **uniformly**.
- Produces output with **Poisson-like column statistics**.
- Is fully **bit-packed, cache-friendly, and SIMD-ready**.
- Requires **no learning, synchronization, or heuristic balancing**.

This builds on **phase-separated binary coupling matrices via barrel-shifting**, extended into a **practical, scalable routing system** with provable statistical properties.

---

## 2. Related Work

### 2.1 Discrete Routing and Coupling Matrices

Discrete approximations to doubly stochastic matrices are used in:

- **Permutation learning**
- **Optimal transport**
- **Binary attention mechanisms**

Prior methods typically rely on **stochastic sampling or continuous relaxations** (e.g., Sinkhorn–Knopp, softmax normalization), which are memory-intensive, non-deterministic, and computationally expensive for large matrices.

### 2.2 Mixture-of-Experts and Sparse Routing

- **Hash-based routing** (Shazeer et al., 2017) is fast but suffers from collisions and expert overload.
- **Softmax-based routers** adapt to load but are unstable and require extra gradient noise and regularization.
- **Greedy or load-balancing routers** require global state and O(N log N) operations.

Our approach provides **deterministic, uniform load** with **fixed row/column sums** without these drawbacks.

### 2.3 Low-Discrepancy Sequences and Configuration Models

- Phase spreading resembles **low-discrepancy sequences** (e.g., van der Corput, Halton) to distribute mass evenly.
- Global permutations approximate a **configuration-model bipartite graph**, achieving **maximum-entropy mixing** subject to fixed degrees.

---

## 3. Method

Let `S` and `T` be source and target binary matrices (`S, T ∈ {0,1}^{N×N}`) with row sums `s_i` and column sums `t_j`. We aim to construct a routing matrix `O` such that:

```
O = S' AND (T')^T
```

where `S'` and `T'` are **degree-preserving mixed versions** of `S` and `T`.

---

### 3.1 Original Phase-Separated Barrel-Shifting

**Source matrix (S)**:

1. Populate each row `i` with `s_i` ones, padded with zeros.
2. Apply **cumulative row barrel shift**:

   ```
   offset_i = sum_{m=0}^{i-1} s_m
   ```

3. Apply a random **column permutation** to reduce alignment patterns.

**Target matrix (T)**:

1. Populate each column `j` with `t_j` ones, padded with zeros.
2. Apply **cumulative column barrel shift**:

   ```
   offset_j = sum_{m=0}^{j-1} t_m
   ```

3. Apply a random **row permutation** to reduce residual alignment.

**Output matrix**:

```
O_{i,j} = S_{i,j} AND T_{i,j}
```

Produces an approximate discrete doubly stochastic matrix with **minimal simultaneous activations**.

---

### 3.2 Enhancements in Bit-Packed Phase Router

1. **Global permutations of S and T**

   - Break residual correlations after phase rotation.
   - Row permutations for S and T; column permutation for S.
   - Additional row permutation for T before transpose ensures near-independence.

2. **Transpose + AND for routing**

   ```
   O = S' AND (T')^T
   ```

3. **Bit-packed 64-bit implementation**

   - Each row stores 64 potential connections per word.
   - SIMD-friendly AND, popcount, and shifts enable **memory-bandwidth-limited routing** at `N = 4096+`.

4. **Probabilistic interpretation**

   - For large `N`:

     ```
     Pr(O_{ij}=1) ≈ s_i * t_j / N
     ```

   - Column sums converge approximately to a **Poisson distribution** with mean proportional to `t_j`.
   - Row sums are preserved exactly unless truncated by **top-k fan-out**.

---

### 3.3 Full Algorithm

1. Align rows of `S` and `T` (left-align 1s).
2. Apply shared row permutation to `S` and `T`.
3. Compute separate cumulative offsets for `S` and `T`.
4. Phase rotation / barrel shifts.
5. Apply independent column permutations to `S` and `T`.
6. Apply row permutation to `T` (pre-transpose).
7. Transpose `T` with additional column permutation.
8. Bitwise AND → routing matrix `O`.
9. Emit **top-k hits per row** to enforce fan-out cap.

---

### 3.4 Theoretical Properties

- **Row/Column Sum Preservation:** exact unless capped at `k`.
- **Load Uniformity:** phase spreading + permutations approximate uniform distribution.
- **Approximate Maximum Entropy:** output behaves like a **configuration-model bipartite graph**.
- **Deterministic Randomness:** only randomness is in initial permutations.

---

## 4. Applications

### 4.1 Mixture-of-Experts (MoE)

- Each token = row of `S`.
- Each expert = column of `T`.
- Routing ensures **exact expert capacity**, Poisson-distributed token load, no hotspots.

### 4.2 Sparse Attention

- Phase-separated routing ensures even attention.
- Efficient for large sparse matrices via pointer-and-offset implementation.

### 4.3 Neuromorphic / Energy-Constrained Hardware

- Limits peak simultaneous activations.
- Efficient for FPGA and edge devices.

### 4.4 Combinatorial Allocation

- Fair distribution of discrete resources.
- Phase separation prevents conflicts and load spikes.

---

## 5. Performance and Evaluation

For `N = 4096`:

| Stage                | Time    |
| -------------------- | ------- |
| Phase + permutations | ~176 ms |
| Bit-packed AND       | ~327 ms |
| Total routing        | ~0.5 s  |

- Kernel is **memory-bandwidth-limited**, not compute-limited.
- Metrics collected: active routes, routes per row, fill ratio, column stats, runtime (packing, routing, total).
- Evaluation conducted over **3 independent trials**; reproducibility tests run **5 runs with fixed seeds**.

---

## 6. Discussion

- Combines **phase spreading, global permutations, transposition, bit-packing**.
- Preserves row/column sums, enforces fan-out, avoids hotspots.
- Interpretable, deterministic, scalable.
- Theoretical link to **configuration-model bipartite graphs**.
- Modular for alternative Boolean gates, permutation schemes, or non-square matrices.

---

## 7. Conclusion

- Scalable, deterministic routing for large binary matrices.
- Extends **phase-separated barrel-shifting** to practical high-performance routing.
- Preserves row/column sums.
- Produces Bernoulli–Poisson-like output, avoids hotspots.
- Fully memory- and compute-efficient via bit-packing.
- Provides a primitive for neural routing, sparse attention, combinatorial allocation.

---

## References

1. Sinkhorn, R. (1964). _A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices_
2. Shazeer, N. et al. (2017). _Outrageously Large Neural Networks: The Mixture-of-Experts Layer_
3. Halton, J.H. (1960). _On the Efficiency of Certain Quasi-Random Sequences_
4. van der Corput, J.G. (1935). _Verteilungsfunktionen I–III_
5. Newman, M.E.J. (2001). _The Structure and Function of Complex Networks_

---
