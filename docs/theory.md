# **Phase-Separated Binary Coupling via Deterministic Mixing**

## Abstract

We present a deterministic construction that maps two binary matrices with prescribed row and column sums into a mixed bipartite coupling whose local statistics converge to those of a random **configuration-model (Chung–Lu)** bipartite graph. The construction uses phase-based dispersal followed by global permutations and supports an efficient bit-packed implementation with bounded fan-out.

---

## 1. Problem

Let

[
S,T\in{0,1}^{N\times N}
]

with row sums

[
s_i=\sum_j S_{ij}, \qquad
t_j=\sum_i T_{ij}.
]

We wish to construct a routing matrix (O) such that

- total flow from source row (i) scales with (s_i),
- total flow into target column (j) scales with (t_j),
- all other structure is maximally mixed.

This corresponds to sampling from a **maximum-entropy bipartite graph with fixed expected degrees**, i.e. the **Chung–Lu / configuration model**, where

[
\mathbb{E}[O_{ij}]=\frac{s_i t_j}{N}.
]

---

## 2. Phase spreading

After left-aligning rows so that each row’s 1-bits are contiguous, each row (i) is cyclically shifted by

[
\phi_i=\sum_{r<i} s_r.
]

This assigns every 1-bit a global **phase** on a ring of size (N) and wraps it around the columns.

As a result, row (i) occupies a contiguous arc of length (s_i) on the phase ring, producing a **low-discrepancy, equidistributed placement** of mass across columns while preserving row sums.

The offsets are accumulated in a randomly permuted row order (see below), so these arcs behave like random intervals on the phase circle.

---

## 3. Global permutations

A shared global permutation of rows is applied to both (S) and (T), followed by independent column permutations.

These operations preserve

- row sums,
- column sums,

while destroying geometric and index-based correlations introduced by alignment and phase spreading.

Before transposition, an **additional independent row permutation** is applied to (T), ensuring that (S) and (T) become **independently mixed degree-preserving fields** in the shared phase space.

---

## 4. Intersection model

After transposing the mixed (T), routing is computed as

[
O_{ij}=S'*{ij}\wedge T'*{ji}.
]

For any fixed pair ((i,j)), this is equivalent to

[
O_{ij}=\sum_{k=1}^N S'*{ik}T'*{jk}.
]

After phase spreading and independent permutations,

[
\Pr(S'*{ik}=1)=\frac{s_i}{N}, \qquad
\Pr(T'*{jk}=1)=\frac{t_j}{N},
]

and these events are asymptotically independent across (k). Therefore

[
\mathbb{E}[O_{ij}]
=\sum\_{k=1}^N
\frac{s_i}{N}\frac{t_j}{N}
=\frac{s_i t_j}{N}.
]

Thus each potential edge behaves as a Bernoulli variable with probability
(p\_{ij}=s_i t_j/N), matching the **Chung–Lu / configuration-model limit**.

---

## 5. Poisson limit

Define the total load on column (j) as

[
O_j=\sum_i O_{ij}.
]

Since (O\_{ij}) is a sum of many small, weakly dependent Bernoulli variables,

[
\mathbb{E}[O_j]
=\sum_i \frac{s_i t_j}{N}
=\frac{t_j}{N}\sum_i s_i
=\frac{|S|,t_j}{N},
]

where (|S|=\sum_i s_i) is the total mass in (S).

When degrees are well spread,

[
O_j\sim \text{Poisson}!\left(\frac{|S|,t_j}{N}\right).
]

If the target degrees (t_j) are approximately uniform, this further simplifies to

[
O_j\sim \text{Poisson}!\left(\frac{|S|,|T|}{N^2}\right).
]

This Poisson behavior explains the absence of hotspots and the uniform “starfield” appearance of the routing matrix.

---

## 5.5 Fan-out truncation

In the implementation, each source row emits at most (k) matches.

Let (\tilde O\_{ij}) denote the uncapped Chung–Lu variable from Section 4.
The realized routing is

[
O_{ij}=\tilde O_{ij};\mathbf{1}!\left(\sum_j \tilde O_{ij}\le k\right),
]

i.e. only the first (k) matches per row are kept.

Thus the row degree distribution satisfies

[
\deg(i)\sim \min!\big(\text{Poisson}(s_i),,k\big),
]

and column loads remain Poisson-like but are mildly truncated by this per-row capacity constraint.

This enforces bounded fan-out while preserving Chung–Lu statistics in the bulk.

---

## 6. Interpretation

The full pipeline implements a **measure-preserving mixing transform** on binary matrices:

- Phase spreading embeds row mass uniformly in a shared cyclic phase space,
- Global permutations eliminate residual structure and induce independence,
- The AND operation samples the product measure,
- Fan-out truncation enforces capacity constraints.

The result is a **deterministic, bit-parallel approximation to a random bipartite configuration model** with bounded degrees.

---

## 7. Why this is useful

This construction produces

- uniform load,
- no hotspots,
- no feedback loops,
- degree-proportional routing with bounded fan-out,

without learning, hashing, greedy balancing, or coordination.

It is therefore well suited for large-scale MoE routing, sparse attention, and other high-throughput bipartite coupling problems.

---

## Monte-Carlo Interpretation of the Phase Router

The bit-packed phase router does **not** attempt to enumerate all valid edges between the bipartite sets defined by `S` and `T`. Instead, it implements a **Chung–Lu–style stochastic sampler** that produces a sparse, balanced bipartite subgraph by sampling a fixed number of edges per row.

This means the router should be understood as a **random transport operator**, not a deterministic solver.

### Chung–Lu Model

Let

- ( S_i ) be the degree (bit count) of row ( i ) in ( S )
- ( T_j ) be the degree (bit count) of column ( j ) in ( T )

The target distribution is a Chung–Lu bipartite graph:

[
P(i \rightarrow j) ;\propto; S_i , T_j
]

subject to a sparsity constraint that each row ( i ) emits at most ( k ) edges.

Each invocation of the router samples a random bipartite graph ( G ) from this distribution, producing a routing matrix

[
O_{ij}^{(s)} \in {0,1}, \quad \sum_j O_{ij}^{(s)} \le k
]

where ( s ) indexes the random seed (sample).

---

### What the Router Actually Outputs

The primary observable of interest is **column load**:

[
L_j^{(s)} = \sum_i O_{ij}^{(s)}
]

This is the number of tokens, queries, or messages routed to column (expert) ( j ) in sample ( s ).

The router produces a **Monte-Carlo sample** of the random vector

[
\mathbf{L}^{(s)} = (L_1^{(s)}, \dots, L_N^{(s)})
]

rather than a fixed routing plan.

Routes themselves are ephemeral; only the induced load distribution matters for performance, balance, and capacity planning.

---

### Expected Loads (Chung–Lu Baseline)

Ignoring the row-cap constraint, the Chung–Lu model predicts

[
\mathbb{E}[L_j] = k \cdot \frac{T*j}{\sum*\ell T\_\ell}
]

The phase router implements this distribution **approximately**, with deviations caused by:

- finite ( k )
- row capacity constraints
- permutation heuristics
- collision avoidance

Monte-Carlo sampling is therefore the correct way to characterize its behavior.

---

### Monte-Carlo Estimation

Given ( S, T, k ), we run the router with multiple independent seeds:

[
\mathbf{L}^{(1)}, \mathbf{L}^{(2)}, \dots, \mathbf{L}^{(M)}
]

From these samples we estimate:

- Mean load
  [
  \mu_j = \frac{1}{M} \sum_s L_j^{(s)}
  ]

- Variance
  [
  \sigma_j^2 = \operatorname{Var}(L_j^{(s)})
  ]

- Tail risk
  [
  Q_{p,j} = \text{p-th percentile of } L_j^{(s)}
  ]

These quantities directly determine:

- expert utilization
- overflow probability
- capacity headroom

---

### Global Load Imbalance (MoE Metric)

For Mixture-of-Experts and attention routing, the key metric is

[
\text{Global Skew}
= \frac{\max_j \mu_j}{\frac{1}{N}\sum_j \mu_j}
]

This measures how overloaded the hottest expert is compared to the average.

A perfectly balanced router has skew = 1.
Typical MoE systems require skew ≤ 1.5–2 to avoid stragglers.

---

### Capacity Planning

Given Monte-Carlo samples, we can estimate expert capacity requirements:

[
C_j(p) = \text{p-th percentile of } L_j^{(s)}
]

To guarantee no overload with probability ≥ ( p ), each expert must support at least ( C_j(p) ) tokens.

This enables statistically principled sizing of MoE experts and communication buffers.

---

### Role of `router_stats.py`

The module `router_stats.py` implements these Monte-Carlo estimators:

| Function                   | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| `sample_many`              | Draws ( \mathbf{L}^{(s)} ) samples          |
| `monte_carlo_stats`        | Estimates mean, variance, percentiles, skew |
| `suggest_k_for_balance`    | Finds minimal ( k ) achieving target skew   |
| `estimate_expert_capacity` | Computes tail-risk capacity                 |

These utilities do **not** modify the router.
They treat it as a black-box sampler and extract the statistically meaningful quantities required for system design.

---
