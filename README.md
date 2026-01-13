# **Bit-Packed Phase Router**

A **high-performance C++ / Python library** for building **balanced, collision-free bipartite routings** between large sparse binary matrices.

It is designed for:

- Mixture-of-Experts (MoE)
- sparse attention
- load-balanced fan-out
- large bipartite graph coupling
- stochastic routing at scale

The router converts two fixed-degree binary matrices into a **uniformly mixed routing field** with preserved row and column sums and **Poisson-like collision statistics**.

All computation is **bit-packed** (64 bits per word) and uses only:

```
AND, shifts, popcount, and permutations
```

making it extremely fast and cache-efficient.

---

## What it computes

Given:

- `S ∈ {0,1}^{N×N}` (sources)
- `T ∈ {0,1}^{N×N}` (targets)

the router produces a routing matrix:

```
O = S' ∧ T'
```

where `S'` and `T'` are **degree-preserving mixed versions** of the inputs.

The output satisfies:

- Row sums of `O` are proportional to row sums of `S`
- Column sums of `O` are proportional to column sums of `T`
- Collisions are spread uniformly
- No column becomes a hotspot

---

## Why this is useful

In many systems (MoE, sharded KV stores, sparse attention), we need to assign many sources to many targets while:

- respecting capacity
- avoiding hotspots
- avoiding feedback loops
- keeping runtime cost low

Most approaches use:

- hashing
- softmax routers
- greedy load balancing

All of these either:

- produce unstable load,
- require learning,
- or require expensive coordination.

The Phase Router instead builds a **degree-preserving random-like bipartite graph by construction**, using only deterministic transforms and permutations.

---

## High-level algorithm

### 1. Align

Rows of `S` and columns of `T` are packed so that all 1s are contiguous.

This preserves all row and column sums.

---

### 2. Global row permutation

Apply a shared row permutation to both `S` and `T` to break initial correlations.

---

### 3. Phase spread with separate offsets

Each matrix gets its own cumulative offsets:

- `S`: offset[i] = sum of all 1s in previous S rows
- `T`: offset[i] = sum of all 1s in previous T rows

Each row is then cyclically rotated by its respective offset.

This distributes each row's mass evenly across columns while maintaining independent phase separation for S and T.

---

### 4. Column permutations

Apply independent column permutations to `S` and `T`:

- preserve all marginals
- destroy geometric structure
- mix phases uniformly

---

### 5. Transpose with enhanced mixing

`T` is transposed with an additional column permutation applied during the transpose operation.

This breaks residual geometric patterns and reduces skew.

---

### 6. Bitwise AND

Routing is computed as:

```
O = S' ∧ T'^T
```

done in bit-packed form using SIMD-friendly ANDs.

---

## What the output looks like

Statistically, the output behaves like a **random bipartite graph with fixed degrees**:

- Each output cell behaves approximately like a Bernoulli random variable
- Each output column behaves approximately like a Poisson random variable

This produces:

- no stripes
- no clusters
- no hotspots

Visually, the matrix looks like a **starfield**.

---

## Performance

For `N = 4096`:

| Stage                | Time    |
| -------------------- | ------- |
| Phase + permutations | ~176 ms |
| Bit-packed AND       | ~327 ms |
| Total routing        | ~0.5 s  |

This is **memory-bandwidth limited**, not compute-limited.

---

## Python API

For most users:

```python
stats = router.pack_and_route(S, T, k, routes)
```

This runs the full pipeline:

```
align → phase → row_perm → col_perm → transpose → AND
```

### Reproducible routing

For deterministic, reproducible routing (e.g., for testing or debugging):

```python
stats = router.pack_and_route(S, T, k, routes, seed=42)
```

With a fixed seed, identical input matrices will always produce identical routes.

Default behavior (`seed=0`) uses time-based seeding for non-deterministic routing.

### Advanced API

Advanced users can provide their own packed arrays and permutations via:

```python
route_packed_with_stats(...)
```

---

## What this is (and is not)

This is:

- a deterministic degree-preserving mixing operator
- a fast way to generate random-like bipartite graphs
- a scalable routing primitive

This is **not**:

- a learned router
- a greedy load balancer
- a heuristic hash

The theoretical construction is documented in `theory.md`.

---
