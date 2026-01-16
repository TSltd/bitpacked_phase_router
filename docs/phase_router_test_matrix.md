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
