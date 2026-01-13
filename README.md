# **Bit-Packed Phase Router**

A **high-performance C++ / Python library** for building **balanced, collision-free bipartite routings** between large sparse binary matrices.

It implements a **deterministic, bit-parallel sampler** for a **degree-capped Chung–Lu bipartite graph** using cyclic phase mixing and permutations.

Designed for:

- Mixture-of-Experts (MoE)
- sparse attention
- load-balanced fan-out
- large bipartite graph coupling
- stochastic routing at scale

All computation is **bit-packed** (64 bits per word) and uses only

```
AND, shifts, popcount, and permutations
```

making it memory-bandwidth limited, cache-efficient, and SIMD-friendly.

---

## What it computes

Given two binary matrices

[
S,T\in{0,1}^{N\times N}
]

with row sums

[
s_i=\sum_j S_{ij}, \qquad
t_j=\sum_i T_{ij},
]

the router constructs up to **k routes per row** by computing

[
O = S' ;\wedge; (T')^{\top}
]

where (S') and (T') are **independently phase-mixed and permuted degree-preserving transforms** of the inputs.

For large (N),

[
\mathbb{E}[O_{ij}] ;\approx; \frac{s_i,t_j}{N}.
]

Thus the router samples a **Chung–Lu (configuration-model) bipartite graph** with:

- larger (s_i) → more outgoing routes
- larger (t_j) → more incoming load

subject to a **hard fan-out cap (k) per source row**.

---

## Guarantees

The output satisfies:

- expected row sums scale with (s_i)
- expected column sums scale with (t_j)
- collisions are uniformly distributed
- no column becomes a geometric or phase-aligned hotspot
- fan-out per row is bounded by (k)

The degrees are preserved **in expectation** (as in a configuration model), not exactly per realization.

---

## Why this is useful

In MoE, sparse attention, and distributed routing, we need to map many sources to many targets while:

- respecting capacity
- avoiding hotspots
- avoiding feedback loops
- keeping runtime cost low

Most routers rely on:

- hashing
- learned softmax gates
- greedy balancing

These are unstable, require training, or need coordination.

The Phase Router instead produces a **random-like bipartite graph by construction**, using only deterministic transforms and bitwise operations.

---

## High-level algorithm

### 1. Align

Rows of `S` and `T` are left-aligned so all 1-bits are contiguous.
This preserves row sums.

---

### 2. Global row permutation

A shared random row permutation is applied to both `S` and `T`.

This determines the order in which mass is placed on the phase ring and removes any input ordering structure.

---

### 3. Phase spreading (independent for S and T)

For permuted row (i), compute cumulative offsets

[
\phi_i^S=\sum_{r<i}s_r, \qquad
\phi_i^T=\sum_{r<i}t_r.
]

Each row is cyclically rotated by its offset, embedding its mass as a contiguous arc on a ring of size (N).

Because offsets are accumulated in **random row order**, these arcs behave like random intervals on a circle.

---

### 4. Independent column permutations

Apply independent column permutations to `S` and `T`.

These preserve column sums while destroying all geometric and phase correlations.

---

### 5. Extra row permutation for T and transpose

Before transposing `T`, apply an **independent row permutation** and an additional column permutation during the transpose.

This makes `S` and `T` statistically independent in the shared phase space while preserving their degree sequences.

---

### 6. Bit-parallel intersection

Compute

[
O_{ij}=\sum_k S'*{ik},T'*{jk}
]

using bit-packed AND and popcount.

From each row, emit the **first (k) hits** in bit order; additional matches are discarded.
This enforces a **hard fan-out limit** per source.

---

## Statistical behavior

After mixing,

[
\Pr(S'*{ik}=1)\approx\frac{s_i}{N},\qquad
\Pr(T'*{jk}=1)\approx\frac{t_j}{N}.
]

Therefore,

[
\mathbb{E}[O_{ij}]\approx\frac{s_i t_j}{N},
]

which is exactly the **Chung–Lu / configuration-model law**.

Column loads

[
O_j=\sum_i O_{ij}
]

are approximately

[
O_j\sim\text{Poisson}!\left(\frac{|S|,t_j}{N}\right),
]

until truncated by the per-row cap (k).

This yields:

- no stripes
- no clusters
- no hotspots

Visually, the output looks like a **starfield**.

---

## Performance

For (N=4096):

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

Advanced users can provide their own packed arrays and permutations via

```python
route_packed_with_stats(...)
```

---

## What this is (and is not)

This is:

- a deterministic degree-weighted mixing operator
- a fast Chung–Lu bipartite sampler
- a scalable routing primitive with hard fan-out limits

This is **not**:

- a learned router
- a greedy load balancer
- a hash

The theoretical construction is documented in [`theory.md`](theory.md)._
Empirical performance and load-balance results are documented in [`evaluation.md`](evaluation.md)._

---
