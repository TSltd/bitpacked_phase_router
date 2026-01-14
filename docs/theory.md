# **Phase-Separated Binary Coupling via Deterministic Mixing**

## Abstract

We present a deterministic construction that maps two binary matrices with prescribed row and column sums into a mixed bipartite coupling whose local statistics match the first-order expected-degree statistics of a Chung–Lu bipartite graph, but the edge dependencies are structured, not independent. The construction uses phase-based dispersal followed by global permutations and supports an efficient bit-packed implementation with bounded fan-out. While Monte-Carlo statistics are used to evaluate robustness, the construction is deterministic given a seed; randomness enters only through phase embeddings.

---

# **Equation Reference Table**

| Symbol / Formula                                | Meaning / Role                                                            | Code / Context Reference                    |                                                 |                        |
| ----------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------- | ---------------------- |
| `S, T ∈ {0,1}^{N×N}`                            | Input binary matrices, indicating allowed sources (`S`) and targets (`T`) | Input to `phase_router.py`                  |                                                 |                        |
| `s_i = sum_j S_{ij}`                            | Row sum of `S` (source load for row `i`)                                  | `row_sums(S)`                               |                                                 |                        |
| `t_j = sum_i T_{ij}`                            | Column sum of `T` (target load for column `j`)                            | `col_sums(T)`                               |                                                 |                        |
| `φ_i = sum_{r < i} s_r`                         | Phase offset for row `i` (cyclic shift to spread 1-bits)                  | Internal in phase spreading step            |                                                 |                        |
| `O_{ij} = sum_k (S'_{ik} ∧ T'_{jk})`            | Routing matrix; counts intersections of mixed matrices                    | Output of `phase_router.py`                 |                                                 |                        |
| `E[O_{ij}] = s_i * t_j / N`                     | Expected flow between source `i` and target `j`                           | Chung–Lu baseline                           |                                                 |                        |
| `O_j = sum_i O_{ij}`                            | Total load on column `j`                                                  | Column load calculation (`router_stats.py`) |                                                 |                        |
| `O_j ~ Poisson(                                 | S                                                                         | \* t_j / N)`                                | Asymptotic Poisson approximation of column load | Monte-Carlo evaluation |
| `O_{ij} = min(tilde O_{ij}, k)`                 | Row fan-out truncation to enforce bounded capacity                        | `phase_router.py` implementation            |                                                 |                        |
| `deg(i) ~ min(Poisson(s_i), k)`                 | Effective row degree distribution after truncation                        | For analysis / expected behavior            |                                                 |                        |
| `L_j^{(s)} = sum_i O_{ij}^{(s)}`                | Column load for Monte-Carlo sample `s`                                    | `sample_many()`                             |                                                 |                        |
| `μ_j = (1/M) * sum_s L_j^{(s)}`                 | Mean column load across Monte-Carlo samples                               | `monte_carlo_stats()`                       |                                                 |                        |
| `σ_j^2 = Var(L_j^{(s)})`                        | Variance of column loads                                                  | `monte_carlo_stats()`                       |                                                 |                        |
| `Q_{p,j}`                                       | p-th percentile of column load                                            | `estimate_expert_capacity()`                |                                                 |                        |
| `Global Skew = max_j μ_j / ((1/N) * sum_j μ_j)` | Load imbalance metric for MoE routing                                     | `suggest_k_for_balance()`                   |                                                 |                        |

---

## Assumptions

The construction assumes that row and column degrees are “well spread” and not adversarial.

---

## 1. Problem

Let

```
S, T ∈ {0,1}^{N × N}
```

with row sums

```
s_i = sum_j S_{ij},   t_j = sum_i T_{ij}.
```

We wish to construct a routing matrix `O` such that:

- total flow from source row `i` scales with `s_i`,
- total flow into target column `j` scales with `t_j`,
- all other structure is maximally mixed.

This corresponds to sampling from a **maximum-entropy bipartite graph with fixed expected degrees**, i.e., the **Chung–Lu / configuration model**, where

```
E[O_{ij}] = (s_i * t_j) / N.
```

---

## 2. Phase Spreading

After left-aligning rows so that each row’s 1-bits are contiguous, each row `i` is cyclically shifted by

```
φ_i = \sum_{m < i} s_{\pi(m)}

```

where π is a fixed, seed-dependent permutation of rows.

This assigns every 1-bit a global **phase** on a ring of size `N` and wraps it around the columns.

As a result, row `i` occupies a contiguous arc of length `s_i` on the phase ring, producing a **low-discrepancy, equidistributed placement** of mass across columns while preserving row sums.

Offsets are accumulated in a fixed but seed-dependent permuted row order. This induces quasi-random, low-discrepancy phase placements rather than independent random intervals.

---

## 3. Global Permutations

A shared global permutation of rows is applied to both `S` and `T`, followed by **independent column permutations**. These operations preserve:

- row sums,
- column sums,

while destroying geometric and index-based correlations introduced by alignment and phase spreading.

Before transposition, an **additional independent row permutation** is applied to `T`, ensuring that `S` and `T` become **independently mixed degree-preserving fields** in the shared phase space.

---

## 4. Intersection Model

After transposing the mixed `T`, routing is computed as:

```
O_{ij} = sum_{k=1}^{N} (S'_{ik} ∧ T'_{jk})
```

For any fixed pair `(i,j)`, this is equivalent to a **sum of weakly dependent Bernoulli variables**. Correlations are small due to low-discrepancy phase packing and random permutations. After phase spreading and independent permutations:

```
Pr(S'_{ik} = 1) = s_i / N,   Pr(T'_{jk} = 1) = t_j / N
```

and these events are asymptotically independent across `k`. Therefore:

```
E[O_{ij}] = sum_{k=1}^{N} (s_i / N) * (t_j / N) = (s_i * t_j) / N
```

Thus each potential edge behaves as a Bernoulli variable with probability

```
p_{ij} = (s_i * t_j) / N,
```

matching the **Chung–Lu / configuration-model limit**.

---

## 5. Poisson Limit

Define the total load on column `j` as:

```
O_j = sum_i O_{ij}.
```

Since `O_{ij}` is a sum of many small, weakly dependent Bernoulli variables:

```
E[O_j] = sum_i (s_i * t_j) / N = (t_j / N) * sum_i s_i = (|S| * t_j) / N
```

where `|S| = sum_i s_i` is the total mass in `S`.

When degrees are well spread:

```
O_j ~ Poisson(|S| * t_j / N)
```

If the target degrees `t_j` are approximately uniform:

```
O_j ~ Poisson(|S| * |T| / N^2)
```

This approximation holds for large `N` and moderate degrees; truncation by `k` introduces slight negative correlations.

This Poisson behavior explains the absence of hotspots and the uniform “starfield” appearance of the routing matrix.

---

## 6. Fan-Out Truncation

In the implementation, each source row emits at most `k` matches. Let `tilde O_{ij}` denote the uncapped Chung–Lu variable from Section 4. For each row `i`, keep only the first `k` intersections; additional matches are discarded:

```
O_{ij} = min(tilde O_{ij}, k)   where sum_j O_{ij} ≤ k
```

or equivalently:

```
O_{ij} = min(tilde O_{ij}, k)
```

Thus the row degree distribution satisfies:

```
deg(i) ~ min(Poisson(s_i), k)
```

and column loads remain Poisson-like but are mildly truncated by this per-row capacity constraint.

This enforces **bounded fan-out** while preserving Chung–Lu statistics in the bulk.

---

## 7. Interpretation

The full pipeline implements a **measure-preserving mixing transform** on binary matrices:

1. **Phase spreading** embeds row mass uniformly in a shared cyclic phase space.
2. **Global permutations** eliminate residual structure and induce independence.
3. The **AND operation** samples the product measure.
4. **Fan-out truncation** enforces capacity constraints.

Result: a **deterministic, bit-parallel approximation to a random bipartite configuration model** with bounded degrees.

---

## 8. Why This is Useful

This construction produces:

- uniform load,
- no hotspots,
- no feedback loops,
- degree-proportional routing with bounded fan-out,

without learning, hashing, greedy balancing, or coordination. Ideal for **large-scale MoE routing**, **sparse attention**, and other **high-throughput bipartite coupling problems**.

---

### Monte-Carlo Interpretation of the Phase Router

The bit-packed phase router implements a **Chung–Lu–style stochastic sampler**:

- **Input:** `(S, T, k)`
- **Output:** Random sparse bipartite graph `O^{(s)}` per seed `s`

```
O_{ij}^{(s)} ∈ {0,1},   sum_j O_{ij}^{(s)} ≤ k
```

Routes are ephemeral; **only induced column loads matter**:

```
L_j^{(s)} = sum_i O_{ij}^{(s)}
```

Collecting `M` independent samples:

```
L^{(1)}, L^{(2)}, ..., L^{(M)}
```

we can estimate:

- Mean: `μ_j = (1/M) * sum_s L_j^{(s)}`
- Variance: `σ_j^2 = Var(L_j^{(s)})`
- Tail risk: `Q_{p,j} = p-th percentile of L_j^{(s)}`

---

### Expected Loads (Chung–Lu Baseline)

Ignoring the row-cap constraint:

```
E[L_j] = k * t_j / sum_l t_l
```

The phase router approximates this distribution, with deviations due to:

- finite `k`
- row capacity constraints
- permutation heuristics
- collision avoidance

Monte-Carlo sampling is the **correct way** to characterize its behavior.

---

### Global Load Imbalance (MoE Metric)

```
Global Skew = max_j μ_j / ((1 / N) * sum_j μ_j)
```

- Perfectly balanced: skew = 1
- Typical MoE systems: skew ≤ 1.5–2 to avoid stragglers

---

### Capacity Planning

Given Monte-Carlo samples, estimate expert capacity:

```
C_j(p) = p-th percentile of L_j^{(s)}
```

Guarantee no overload with probability ≥ `p`.

---

### Flow Diagram (Monte-Carlo Sampling)

```
S, T, k, seed   ──► Phase Router ──► O^(seed) ──► Column Loads L^(seed)
     │                        │
     │                        └─► Repeat M seeds ─► Monte-Carlo statistics (mean, variance, skew, tail)
```

---

### Role of `router_stats.py`

| Function                   | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| `sample_many`              | Draws L^{(s)} samples                       |
| `monte_carlo_stats`        | Estimates mean, variance, percentiles, skew |
| `suggest_k_for_balance`    | Finds minimal `k` achieving target skew     |
| `estimate_expert_capacity` | Computes tail-risk capacity                 |

These utilities **treat the router as a black-box sampler** and extract statistically meaningful quantities for system design.

---
