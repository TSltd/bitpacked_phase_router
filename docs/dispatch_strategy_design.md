# Dispatch Strategy Design

## Overview

The phase router uses two kernels:

- **original**: robust, permutation-based routing
- **interval-space**: high-performance, density-sensitive routing

A dispatch mechanism selects between them based on input characteristics.

---

## Key Observation: Two Regimes

Empirical benchmarking shows that performance is not smoothly varying across all problem sizes. Instead, the system exhibits a **regime change** around:

```
N ≈ 256
```

### Small-N Regime (N < 256)

- Routing behavior is unstable and highly discrete
- Prefix sums do not sufficiently randomize placement
- Bit-packing and permutation overhead dominate
- Cache effects distort performance
- Kernel performance is inconsistent and noisy

**Conclusion:**
The interval-space kernel does not reliably outperform the original kernel.

---

### Large-N Regime (N ≥ 256)

- Routing becomes statistically stable
- Prefix sums distribute work evenly
- Interval kernel benefits from streaming and locality
- Performance gains become consistent and often large

**Conclusion:**
The interval-space kernel is generally superior except in very dense cases.

---

## Final Dispatch Rule

A simple, robust rule is used:

```cpp
bool use_original;

if (N < 256)
{
    use_original = true;
}
else
{
    double kn = double(k) / double(N);

    // fallback for dense cases
    use_original = (kn > 0.25);
}
```

---

## Why Not Use a Learned Model?

A logistic regression model was explored using features:

- log2(N)
- density
- k/N
- interaction terms

However, it was rejected for the following reasons:

1. **Insufficient training data**
   - Too few (N, k, density) combinations
   - Poor generalization

2. **Distribution mismatch**
   - Synthetic calibration data differed from benchmark inputs

3. **Regime discontinuity**
   - A single continuous model cannot capture the sharp transition at N ≈ 256

4. **Observed regressions**
   - The learned model produced catastrophic mispredictions (up to 10× slowdown)

**Conclusion:**
A simple rule outperforms the learned model in both robustness and portability.

---

## Design Philosophy

- Prefer **robustness over optimality**
- Use **discrete regime boundaries**, not continuous models
- Avoid overfitting to hardware-specific behavior
- Keep dispatch logic simple and predictable

---

## Notes

- The threshold `N = 256` is empirically derived and may vary slightly across systems, but is expected to be stable within the same order of magnitude.
- The density fallback (`k/N > 0.25`) prevents interval kernel degradation in dense cases.

---

## Summary

The dispatch problem is best understood as a **two-regime system**, not a continuous optimization problem. A simple rule based on problem size and density provides reliable, portable performance.
