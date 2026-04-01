# Bitpacked Phase Router — Performance Model & Empirical Findings

## Overview

This document summarizes the performance characteristics of the hybrid bitpacked phase router, based on empirical benchmarking and model fitting.

The system consists of two primary phases:

1. **Routing (interval or original kernel)**
2. **Extraction (bitwise AND + candidate selection)**

We modeled total runtime as a function of:

- Problem size (`N`)
- Input/output density (`f`)
- Derived quantities such as number of active events and memory access patterns

---

## Core Model

The best-performing model takes the form:

```
T ≈ d + a·N + b·(N·f) + c·W
```

Where:

- `d` = constant overhead (Python/C++ boundary, allocation, scheduling)
- `a·N` = routing cost (prefix sums, interval placement)
- `b·(N·f)` = event processing cost (surviving bits)
- `c·W` = memory scanning cost

We approximate:

```
W ≈ N · (1 - (1 - f)^64)
```

This models the probability that a 64-bit word contains at least one active bit.

---

## Key Insight: Three Physical Costs

The algorithm decomposes into three fundamental cost components:

| Component | Description               | Scaling          |
| --------- | ------------------------- | ---------------- |
| Routing   | Prefix sums + placement   | O(N)             |
| Events    | Processing surviving bits | O(N·f)           |
| Scan      | Memory/word traversal     | O(words touched) |

---

## Regime Behavior

### 1. Small Regime (N ≤ 8192)

**Coefficients:**

```
a ≈ 0.0072
b ≈ 0.0060
c ≈ 0.0027
```

**Characteristics:**

- Cache-resident (L2/L3)
- Compute-bound
- All costs similar magnitude

**Model accuracy:**

- Mean error: ~13%
- Median error: ~9%

---

### 2. Large Regime (N ≥ 16384)

**Coefficients:**

```
a ≈ 0.0307
b ≈ 0.0309
c ≈ 0.0059
```

**Characteristics:**

- Memory-bound (DRAM)
- Significant slowdown (~4–5×)
- Event cost dominates

**Model accuracy:**

- Mean error: ~26%
- Median error: ~18%

---

## Critical Observations

### 1. Memory Hierarchy Transition

There is a clear transition between:

- **Cache-resident regime (fast)**
- **Memory-bound regime (slow)**

All cost components increase, but not equally.

---

### 2. Event Cost Dominates at Scale

In the large regime:

```
b >> c
```

Meaning:

> Processing surviving bits is more expensive than scanning memory.

This is due to:

- Random access into `T_routed`
- Cache misses
- Poor locality

---

### 3. Scan Cost is Partially Hidden

Scan cost increases less because:

- Access is sequential
- Hardware prefetching is effective

---

### 4. Runtime is Piecewise Linear

A single global model is insufficient.

Instead:

```
T_small ≠ T_large
```

Performance must be modeled per regime.

---

## Practical Implications

### 1. Dispatch Strategy

Instead of heuristic dispatch:

```cpp
if (kn > threshold)
```

We can use:

```
predict(original_time) vs predict(interval_time)
```

and choose the faster kernel.

---

### 2. Predictive Performance

We can estimate runtime before execution:

```
T_est ≈ d + aN + b(Nf) + cW
```

This enables:

- Scheduling decisions
- Resource planning
- Adaptive algorithms

---

### 3. Optimization Targets

For large N, focus on:

- Reducing random memory access
- Improving locality of `T_routed`
- Blocking / tiling strategies
- Prefetching

---

## Limitations of Current Model

Remaining error (~10–25%) is due to:

- Cache effects not explicitly modeled
- OpenMP scheduling overhead
- Branching differences (sparse vs dense paths)
- Non-uniform bit distributions

---

## Next Step: Instrumentation

To improve accuracy, we will measure real metrics directly in C++:

### Planned instrumentation:

- Number of events processed
- Number of words touched
- Possibly cache-friendly vs random access counts

### Target model:

```
T ≈ a·N + b·events + c·words_touched
```

This should reduce error to ~10% or better.

---

## Summary

We have established:

- A decomposition of runtime into physical cost components
- A regime-dependent performance model
- A strong empirical fit (~10–25% error)
- Clear identification of bottlenecks

This provides a foundation for both:

- Further optimization
- Predictive scheduling and dispatch

---

**Status:** Model validated, ready for instrumentation phase
