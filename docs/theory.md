# **Phase-Separated Binary Coupling via Deterministic Mixing**

## Abstract

We present a deterministic construction that maps two binary matrices with prescribed row and column sums into a mixed bipartite coupling whose local statistics converge to those of a random configuration model. The construction uses phase-based dispersal followed by global permutations and supports an efficient bit-packed implementation.

---

## 1. Problem

Let

[
S, T \in {0,1}^{N \times N}
]

with row sums ( s_i ) and column sums ( t_j ).

We want to produce a routing matrix ( O ) such that:

- total flow from each source row ( i ) scales with ( s_i )
- total flow into each target column ( j ) scales with ( t_j )
- all other structure is maximally mixed

This corresponds to sampling from a **maximum-entropy bipartite graph** under fixed degrees.

---

## 2. Phase spreading

After left-aligning rows, each row ( i ) is cyclically shifted by

[
\phi_i = \sum_{r<i} s_r
]

This assigns every 1-bit a global phase and wraps it around the columns. This produces an equidistributed, low-discrepancy placement of mass.

Each column receives nearly equal mass.

---

## 3. Global permutations

Two global permutations (rows and columns) are applied.

These preserve:

- row sums
- column sums

but destroy all geometric correlations.

A second independent row permutation is applied to (T) before transpose to decorrelate it from (S).

This yields two independently mixed degree-preserving fields.

---

## 4. Intersection model

After transposition,

[
O_{ij} = S'*{ij} \land T'*{ji}
]

For large (N), the probability that a given cell is 1 satisfies

[
\Pr(O_{ij}=1) \approx \frac{s_i t_j}{N^2}
]

subject to the global degree constraints.

This matches the **Chung–Lu / configuration-model limit**.

---

## 5. Poisson limit

Column totals

[
O_j = \sum_i O_{ij}
]

are sums of many small Bernoulli terms. When degrees are well-spread, this converges to

[
O_j \sim \text{Poisson}!\left(\frac{|S|,t_j}{N^2}\right)
]

and if (t_j) is approximately uniform,

[
O_j \sim \text{Poisson}!\left(\frac{|S||T|}{N^3}\right)
]

---

## 6. Interpretation

The full pipeline implements a **measure-preserving mixing transform** on binary matrices:

- Phase spreading enforces uniform marginal dispersion
- Global permutations eliminate residual structure
- The AND operation samples the product measure

The result is a **deterministic approximation to a random bipartite configuration model**.

---

## 7. Why this is useful

This construction produces:

- uniform load
- no hotspots
- no feedback loops
- fixed degrees

without learning, hashing, or coordination.

It is particularly well-suited to large-scale MoE and sparse routing systems.

---
