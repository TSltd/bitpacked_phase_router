# **Phase Router as a Deterministic Hash Mapping Primitive**

## Overview

The Phase Router, originally designed for sparse bipartite routing and load-balanced Mixture-of-Experts (MoE) applications, can be applied as a **deterministic, multi-candidate hash mapping primitive**. In this context, it assigns keys (or items) to buckets (or slots) while maintaining:

- Low skew in bucket load
- Hard limits on fan-out (number of candidate buckets per key)
- Deterministic, seed-reproducible placement
- Optional decorrelation of key order via row permutations

Unlike conventional hash functions, the Phase Router provides **structured, phase-based mappings** rather than purely random placements. This can improve balance in high-throughput or memory-constrained systems.

---

## 1. Conceptual Mapping

### 1.1 Key-to-Bucket Assignment

- Treat each key as a row and each potential bucket as a column in a binary matrix `S`.
- Construct a second matrix `T` to represent target bucket capacities.
- Apply the Phase Router pipeline:
  1. **Phase spreading**: cyclically shift key rows to spread assignments evenly across buckets.
  2. **Column permutations**: shuffle bucket positions to reduce geometric bias.
  3. **Optional row permutations**: anonymize input order if needed.
  4. **Bitwise AND & top-k selection**: intersect matrices and select up to `k` candidate buckets per key.

The result is a **deterministic set of candidate buckets** for each key, suitable for hash table placement or distributed sharding.

### 1.2 Candidate Selection

- Each key may have multiple candidate buckets (`k` candidates).
- At insertion or query time, one can select a bucket using:
  - First available bucket
  - Additional priority metric
  - Round-robin or random deterministic selection based on the seed

This is similar to **multi-choice hashing** or cuckoo hashing, but with controlled skew and deterministic mapping.

---

## 2. Advantages Over Traditional Hash Functions

| Feature                    | Phase Router Hash Mapping                   | Standard Hash Function        |
| -------------------------- | ----------------------------------------- | ---------------------------- |
| Determinism                | Seed-deterministic, reproducible          | Typically deterministic (depends on function) |
| Candidate Buckets          | Multiple per key, up to `k`               | Usually 1                     |
| Load Balance               | Statistically low skew across buckets     | Can produce hotspots          |
| Handling Structured Inputs | Smooth, decorrelated placement            | Sensitive to input patterns   |
| Computational Model        | Batch, bit-packed, cache-efficient        | Scalar, depends on hash       |

---

## 3. Practical Applications

### 3.1 Multi-Choice Hash Tables
- Insert keys into one of multiple candidate buckets
- Reduces collisions and hotspots
- Supports deterministic placement for distributed systems

### 3.2 Distributed Key Assignment / Sharding
- Map keys to nodes or shards with bounded load
- Optional row permutation provides anonymization if input ordering is correlated

### 3.3 Parallel or GPU-Friendly Bucket Assignment
- Batch assign many keys simultaneously using bit-parallel operations
- Suitable for high-throughput packet routing, in-memory caches, or large hash tables

### 3.4 Redundancy and Replication
- Multiple candidate buckets allow replication or redundancy
- Deterministic top-k selection enforces consistent fan-out per key

### 3.5 Memory-Constrained or Real-Time Systems
- Predictable memory access and low-skew mapping
- Sparse, bit-packed representation minimizes storage overhead
- Fully deterministic, no training or randomization at runtime

---

## 4. Example Workflow

```
# Inputs
keys = [k1, k2, ..., kN]
buckets = [b1, b2, ..., bM]

# Construct source and target matrices
S = build_key_matrix(keys, N)
T = build_bucket_matrix(buckets, N)

# Apply Phase Router
O = phase_router(S, T, k, seed=42, row_permute=True)

# Result: candidate buckets for each key
candidates[key_i] = [b_j for j where O[i,j]==1]

# Final selection (single bucket)
selected_bucket = choose_bucket(candidates[key_i])
```

- `k` = maximum candidate buckets per key
- `seed` = ensures deterministic reproducibility
- `row_permute` = optional, decorrelates input order

---

## 5. Summary

Using the Phase Router as a hash mapping primitive enables:

- **Low-skew, deterministic mapping** from keys to buckets
- **Hard fan-out limits** per key
- **Optional anonymization** of input order
- **Efficient batch computation** suitable for high-throughput or distributed applications

It is particularly useful when predictable load distribution and reproducibility are important, and when multiple candidate placements per key are desirable for redundancy or collision avoidance.

