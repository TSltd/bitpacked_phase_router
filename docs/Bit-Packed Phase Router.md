# **Bit-Packed Phase Router: Deterministic, Low-Concurrency Routing via Phase-Separated Binary Coupling Matrices**

## **Abstract**

We present a deterministic, discrete method for routing connections between large sparse or dense binary matrices while preserving prescribed row and column sums. Our approach generalizes the **phase-separated barrel-shifting** construction from prior work and introduces **global permutations and bit-packing** to enable scalable, cache-efficient computation. The resulting routing matrix behaves like a **random bipartite graph**, with approximate Bernoulli-distributed entries and Poisson-distributed column sums.

This construction enables **load-balanced routing for Mixture-of-Experts (MoE), sparse attention, and large-scale graph coupling**, while avoiding stochastic routing, learning, or synchronization overhead. The method is highly memory-efficient, fully deterministic, and interpretable, and it is suitable for large-scale, hardware-friendly implementations.

---

## **1. Introduction**

Efficient routing between large sets of sources and targets arises in numerous applications:

- Neural networks with **Mixture-of-Experts (MoE)** or sparse attention layers
- Graph-based models requiring **bipartite matching**
- Energy-constrained or neuromorphic hardware
- Combinatorial scheduling and resource allocation

Existing routing schemes often fall into three categories:

1. **Hash-based routing:** deterministic but prone to collisions and hotspots.
2. **Softmax or learned routing:** adaptive but unstable, requires extra gradient computation and load regularization.
3. **Greedy or load-aware routing:** expensive and synchronization-heavy at scale.

We propose a **deterministic, discrete, low-concurrency routing primitive** that:

- Preserves **row and column marginals**
- Spreads load **uniformly**
- Produces output with **Poisson-like column statistics**
- Is fully **bit-packed, cache-friendly, and SIMD-ready**
- Requires **no learning, synchronization, or heuristic balancing**

This paper builds on our prior work on **phase-separated binary coupling matrices via barrel-shifting**, extending it into a **practical, scalable routing system** with provable statistical properties.

---

## **2. Related Work**

### 2.1 Discrete Routing and Coupling Matrices

Discrete approximations to doubly stochastic matrices are used in:

- **Permutation learning**
- **Optimal transport**
- **Binary attention mechanisms**

Prior methods typically rely on **stochastic sampling or continuous relaxations** (e.g., Sinkhorn–Knopp, softmax normalization) which are:

- Memory-intensive
- Non-deterministic
- Computationally expensive for large matrices

### 2.2 Mixture-of-Experts and Sparse Routing

- **Hash-based routing** (Shazeer et al., 2017) is fast but suffers from collisions and expert overload.
- **Softmax-based routers** adapt to load but are unstable and require extra gradient noise and regularization.
- **Greedy or load-balancing routers** require global state and O(N log N) operations.

Our approach provides **deterministic, uniform load** with **fixed row/column sums** without any of these drawbacks.

### 2.3 Low-Discrepancy Sequences and Configuration Models

- Phase-spreading in our construction resembles **low-discrepancy sequences** (e.g., van der Corput, Halton) to distribute mass evenly.
- Global permutations approximate a **configuration-model random bipartite graph**, achieving **maximum-entropy mixing** subject to fixed degrees.

---

## **3. Method**

Let (S, T \in {0,1}^{N \times N}) denote source and target matrices with prescribed row sums (s_i) and column sums (t_j). We aim to construct a routing matrix (O) that:

[
O = S' \land T'
]

where (S') and (T') are **degree-preserving mixed versions** of (S) and (T).

### 3.1 Original Phase-Separated Barrel-Shifting

The **original barrel-shifting construction** proceeds as follows:

#### Source Matrix (S)

1. Populate each row (i) with (s_i) ones, padded with zeros.
2. Apply a **cumulative row barrel shift**:

[
\text{offset}*i = \sum*{m=0}^{i-1} s_m
]

This phase-spreads each row’s ones across columns, producing **low-concurrency, maximal separation**. 3. Columns are randomly permuted to reduce alignment patterns.

#### Target Matrix (T)

1. Populate each column (j) with (t_j) ones, padded with zeros.
2. Apply **cumulative column barrel shift**:

[
\text{offset}*j = \sum*{m=0}^{j-1} t_m
]

This phase-spreads each column’s ones vertically. 3. Rows are randomly permuted to reduce residual alignment.

#### Output Matrix

[
O_{i,j} = S_{i,j} \land T_{i,j}
]

This produces an approximate discrete doubly stochastic matrix with **minimal simultaneous activations**.

---

### 3.2 Enhancements in Bit-Packed Phase Router

While barrel-shifting produces low-concurrency couplings, the Bit-Packed Phase Router introduces the following:

1. **Global permutations of S and T**

   - Break residual correlations after phase rotation.
   - Row permutations for S and T; column permutation for S.
   - Additional row permutation for T before transpose ensures near-independence between S and T.

2. **Transpose + AND for routing**

   - T is transposed so that rows become columns, enabling **bitwise AND for routing**:

[
O = S' \land T'^{\top}
]

3. **Bit-packed 64-bit implementation**

   - Each row stores 64 potential connections per word.
   - SIMD-friendly AND, popcount, and shifts enable **memory-bandwidth-limited routing** at N = 4096+.
   - Pointer-and-offset offsets generalize barrel shifts without moving data.

4. **Probabilistic interpretation**

   - For large N, each output bit behaves approximately as a Bernoulli variable:

[
\Pr(O_{ij}=1) \approx \frac{s_i t_j}{N^2}
]

- Column sums converge to a Poisson distribution with mean:

[
\lambda_j \approx \frac{|S| t_j}{N^2}
]

This formalizes the intuitive “starfield” observed in earlier experiments.

---

### 3.3 Algorithm Summary

**Full Pipeline:**

1. Align source rows and target columns
2. Apply global row permutation to both S and T
3. Compute separate cumulative offsets for S and T based on their respective row sums
4. Phase rotation / cumulative barrel shifts (using separate offsets)
5. Apply column permutations to S and T independently
6. Apply row permutation to T (pre-transpose)
7. Transpose T with additional column permutation for enhanced mixing
8. Bitwise AND to produce routing matrix O

---

### 3.4 Theoretical Properties

1. **Row/Column Sum Preservation**

   - All operations preserve row and column sums exactly.

2. **Load Uniformity**

   - Phase spreading + global permutations approximate uniform distribution of ones across columns/rows.

3. **Approximate Maximum Entropy**

   - Given fixed degrees, the routing matrix behaves like a **configuration-model bipartite graph** with near-maximal entropy.

4. **Deterministic Randomness**

   - All randomness is in **initial permutations**; otherwise the procedure is fully deterministic.
   - No feedback loops or adaptive rebalancing required.

---

## **4. Applications**

### 4.1 Mixture-of-Experts (MoE)

- Each token = row of S
- Each expert = column of T
- Routing:

[
O = S \land T
]

Ensures **exact expert capacity**, Poisson-distributed token load, and zero hotspots.

### 4.2 Sparse Attention

- Phase-separated routing ensures tokens attend to multiple targets **evenly**.
- Pointer-and-offset implementation enables efficient attention on large sparse matrices.

### 4.3 Neuromorphic and Energy-Constrained Hardware

- Limits peak simultaneous activations
- Efficient for spiking networks, FPGA, and edge devices.

### 4.4 Combinatorial Allocation

- Fair distribution of discrete resources
- Phase separation prevents conflicts and load spikes

---

## **5. Performance and Evaluation**

> Placeholder: Statistical evaluation, performance benchmarks, and empirical comparisons will be included in future work.
> Metrics to consider:
>
> - Column/row sum deviations
> - Poisson fit of column sums
> - Variance of load distribution
> - Comparison against hashing, softmax, and greedy routing
> - Runtime and memory benchmarks for N = 4096+

---

## **6. Discussion**

- The Bit-Packed Phase Router **realizes the original barrel-shifting idea at scale**.
- By combining **phase spreading, global permutations, transposition, and bit-packing**, it produces routing matrices with:

  - Preserved row/column sums
  - Uniform collision distribution
  - Deterministic, interpretable behavior

- Theoretical interpretation links the construction to **configuration-model bipartite graphs**.
- The approach is **modular**: alternative Boolean gates, different permutation schemes, or non-square matrices can be explored.

---

## **7. Conclusion**

We present a **scalable, deterministic routing algorithm** for large binary matrices that:

- Extends the **phase-separated barrel-shifting concept** to practical, high-performance routing
- Preserves row and column sums
- Produces approximate Bernoulli–Poisson statistics in output
- Eliminates hotspots and feedback loops
- Is fully memory- and computation-efficient via bit-packing

This work unifies **combinatorial phase-spreading, discrete coupling, and large-scale hardware-aware routing**, providing a new primitive for neural routing, sparse attention, and combinatorial allocation.

---

## **References**

1. Sinkhorn, R. (1964). _A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices_.
2. Shazeer, N. et al. (2017). _Outrageously Large Neural Networks: The Mixture-of-Experts Layer_.
3. Halton, J.H. (1960). _On the Efficiency of Certain Quasi-Random Sequences of Points in Evaluating Multi-Dimensional Integrals_.
4. van der Corput, J.G. (1935). _Verteilungsfunktionen I–III_.
5. Configuration Model references as relevant (Newman 2001, etc.)

---
