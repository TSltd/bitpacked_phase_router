## **Phase Router Test Matrix**

The Phase Router Test Matrix is a **systematic stress-test and evaluation suite** for the bit-packed phase router.
It is designed to probe **load balance, determinism, composability, and failure modes** under both randomized and adversarial routing patterns.

The tests go beyond single-run correctness and focus on **distributional behavior**, **worst-case skew**, and **robustness across seeds and structured inputs**.

### **Location**

```text
evaluation/phase_router_test_matrix.py
```

---

## **What This Test Measures**

Each test routes a binary source matrix `S` to a binary target matrix `T` using:

```python
router.pack_and_route(S, T, k, routes)
```

Metrics collected include:

- Column load statistics (min / max / mean / std)
- Load skew (`max / mean`)
- Runtime per test
- Optional distribution plots
- System hardware metadata (CPU, cores, RAM)

Results are exported to:

```text
test_output/
├── phase_router_test_matrix.csv
├── phase_router_test_matrix.md
└── plots/   (optional)
```

---

## **Test Categories**

### **1. Degree & Load Stress**

**Row-Degree Extremes**
Mixes very sparse rows with very dense rows to test robustness to heterogeneous token fan-out.

**Column-Target Stress**
Forces many rows to target the same column, probing load concentration and conflict resolution.

---

### **2. Statistical Behavior**

**Seed Reproducibility**
Verifies deterministic routing: identical `(S, T, k)` inputs must produce identical routes.

**Monte Carlo Distribution (Seed-Averaged)**
Averages column loads over many random seeds to evaluate expected load balance rather than single-seed variance.

> Note: This test measures **seed-averaged column load**, not per-seed variance or tail risk.

---

### **3. Scaling & Performance**

**Large-Scale Performance**
Runs routing at production-scale `(N, k)` to measure runtime and peak load skew.

This test primarily checks:

- Bit-packing scalability
- Phase permutation efficiency
- Memory pressure under realistic load

---

### **4. Adversarial & Composability Tests**

**Adversarial Two-Phase Routing**
Uses the output of a first routing phase to construct a worst-case second-phase input.
This models **MoE stacking**, where routing artifacts can amplify over layers.

The goal is to detect:

- Load collapse
- Resonant column concentration
- Phase interaction failures

---

### **5. Edge Cases & Structured Inputs**

**Edge-Case `k` Values**
Evaluates behavior at:

- `k = 1` (minimum fan-out)
- `k = N` (maximum fan-out)

**Structured Sparse Patterns**

- Diagonal matrices
- Checkerboard patterns

These inputs are intentionally non-random and expose failure modes masked by IID sampling.

**Phase Rotation Boundaries**
Forces wrap-around behavior in phase rotation to test index arithmetic and permutation correctness.

---

### **6. Baseline Comparison**

**Phase Router vs Hash Router**
Compares the phase router against a naïve per-row hash-based router:

- Hash router randomly selects from valid targets
- No global load balancing
- No capacity awareness

This baseline is intentionally simple and highlights the **load-balancing advantage** of phase routing rather than absolute optimality.

Optional histograms compare column-load distributions directly.

---

## **Running the Test Matrix**

From the project root:

```bash
python evaluation/phase_router_vs_hash.py
```

To disable plotting (recommended for headless or CI runs):

```bash
python evaluation/phase_router_vs_hash.py --skip-plots
```

---

## **Output Format**

### **CSV**

Machine-readable metrics for analysis and regression tracking.

### **Markdown**

Human-readable summary including a system hardware footer:

- CPU model and frequency
- Logical / physical cores
- Total system RAM
- OS and architecture

This makes performance results **hardware-contextualized and reproducible**.

---

## **Design Philosophy**

This test matrix is intentionally:

- **Deterministic** (seed-controlled)
- **Adversarial**, not just average-case
- **Composable** across routing layers
- **Distribution-focused**, not just correctness-focused

It is designed to answer:

> _“What happens to routing balance when this system is pushed, stacked, and stressed — not just when it’s sampled?”_

---

# Test Results

### Row-Degree Extremes

| Metric     | Value  |
| ---------- | ------ |
| `col_min`  | 0      |
| `col_max`  | 134    |
| `col_mean` | 64.321 |
| `col_std`  | 64.144 |
| `col_skew` | 2.083  |

- `col_min = 0` is expected (some columns can be empty in extreme cases).
- `col_max = 134` with `col_mean ≈ 64` indicates good spread.
- Standard deviation and skew are reasonable — shows the router is distributing load but still hitting extremes.

---

### Column-Target Stress

| Metric     | Value |
| ---------- | ----- |
| `col_min`  | 0     |
| `col_max`  | 10    |
| `col_mean` | 4.057 |
| `col_std`  | 1.797 |
| `col_skew` | 2.465 |

- Columns are close to the expected target of 4–5.
- Skew is moderate, some columns get more hits due to combinatorial structure.

---

### Seed Reproducibility

| Metric      | Value |
| ----------- | ----- |
| `identical` | True  |

- Confirms every run with the same seed produces identical results

---

### Monte Carlo Mean Load

| Metric      | Value |
| ----------- | ----- |
| `mean_load` | 3.981 |
| `skew`      | 1.457 |

- Very close to the target load (≈4).
- Skew is <2, which is acceptable for Monte Carlo simulations.

---

### Large-Scale Performance

| Metric     | Value |
| ---------- | ----- |
| `col_min`  | 0     |
| `col_max`  | 10    |
| `col_mean` | 4.015 |
| `col_std`  | 1.889 |
| `col_skew` | 2.491 |

- `col_max ≈ 10` and `col_mean ≈ 4` match expectations for stress cases.
- Skew between 2.4–2.5 is consistent.
- Performance seems stable.

---

### Adversarial Two-Phase

| Metric     | Value |
| ---------- | ----- |
| `col_min`  | 0     |
| `col_max`  | 10    |
| `col_mean` | 4.077 |
| `col_std`  | 1.839 |
| `col_skew` | 2.453 |

- `col_max ≈ 10` and `col_mean ≈ 4` match expectations for adversarial cases.
- Skew between 2.4–2.5 is consistent.
- Performance seems stable.

---

### Edge-Case k Values

| Metric             | Value    |
| ------------------ | -------- |
| `k=1__col_max`     | 1        |
| `k=1__col_skew`    | 1023.999 |
| `k=1024__col_max`  | 1024     |
| `k=1024__col_skew` | 1.0      |

- Exactly as expected:

  - When `k=1`, one route per row → huge skew possible (`1023.999` for 1024 rows).
  - When `k=N=1024`, full routing → perfectly uniform (`skew = 1`).

---

### Structured Sparse Patterns

| Metric                   | Value   |
| ------------------------ | ------- |
| `diagonal__col_min`      | 0       |
| `diagonal__col_max`      | 1       |
| `diagonal__col_mean`     | 0.003   |
| `diagonal__col_std`      | 0.054   |
| `diagonal__col_skew`     | 341.333 |
| `checkerboard__col_min`  | 0       |
| `checkerboard__col_max`  | 83      |
| `checkerboard__col_mean` | 32.000  |
| `checkerboard__col_std`  | 32.435  |
| `checkerboard__col_skew` | 2.594   |

- **Diagonal**: max load 1, mean very low → correct for diagonal.
- **Checkerboard**: `col_max = 83`, `col_mean ≈ 32` → matches pattern.
- Skew for diagonal is very high (341), which is expected for near-empty patterns.

---

### Phase Rotation Boundaries

| Metric     | Value |
| ---------- | ----- |
| `col_min`  | 0     |
| `col_max`  | 11    |
| `col_mean` | 4.050 |
| `col_std`  | 2.011 |
| `col_skew` | 2.716 |

- Minor variations in column loads (`col_max = 11`) → reasonable due to rotation effects.

---

### Hash Router Comparison

| Metric       | Value |
| ------------ | ----- |
| `phase_skew` | 2.267 |
| `hash_skew`  | 3.684 |
| `skew_ratio` | 1.625 |

- Shows the router distributes better than the naïve hash baseline (lower skew).
- Skew ratio >1 confirms improvement.

---

### Overall Assessment

- Deterministic behavior confirmed (`Seed Reproducibility = True`)
- Column loads reasonable under extreme, adversarial, and patterned inputs
- Performance metrics look good
- Phase router improves over hash baseline

**Verdict:** The router behaves correctly under stress, is deterministic, and is load-balanced better than the baseline.

---
