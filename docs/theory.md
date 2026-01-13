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
