# **Phase Router Python Test & Run Scripts – Specification**

## **1. Overview**

We have four scripts for testing the phase router:

- **`phase_router_test.py`** – executes **single tests** for a given matrix size (`N`) and routing degree (`k`). Performs correctness checks, statistics, and optional visual outputs.
- **`phase_router_run.py`** – executes **batches of tests** over multiple `N`, `k`, and threading configurations, collects metrics, and produces reproducibility and scaling data.
- **`phase_router_vs_hash.py`** – performs **comprehensive stress tests** comparing the phase router against a simple hash-based router, including single-phase and adversarial two-phase composability tests.
- **`phase_router_test_matrix.py`** – runs a **systematic test matrix** covering load balance, determinism, composability, and failure modes across various input patterns and edge cases.

These scripts are designed to work with the **single-phase router** (`router.router`, `pack_and_route`, `route_packed_with_stats`).

---

## **2. Objectives**

1. **Functional correctness**: ensure routing is valid and deterministic (when using fixed seeds).
2. **Statistical evaluation**: measure column load, row coverage, fill ratio, load balance.
3. **Performance profiling**: measure runtime per stage and scaling with N, k, and threads.
4. **Visual verification**: optional PBM/PNG outputs to verify phase separation and uniform dispersal.
5. **Reproducibility**: fixed-seed runs produce identical results; random-seed runs produce statistically consistent outcomes.

---

## **3. `phase_router_test.py` Specification**

### **3.1 Inputs**

- `N` – matrix size (number of rows/columns).
- `k` – max routes per row.
- `seed` – optional, for reproducibility.
- `dump` – optional flag to save intermediate PBM/PNG images.
- `prepacked` – optional, whether to test pre-packed bit arrays.

### **3.2 Workflow**

1. **Matrix generation**

   - Random binary matrices `S` and `T` with row sums drawn from uniform `[1, k_max]`.
   - Optionally allow fixed patterns for unit tests.

2. **Routing**

   - Call **router API**:

     - `router.router(S, T, k, routes)`
     - or `pack_and_route(S, T, k, routes)`
     - or `route_packed_with_stats(S_bits, T_bits, row_perm, col_perm_S, col_perm_T, row_perm_T, k, routes)` for pre-packed tests.

   - Optional multi-trial with fixed/random seeds.

3. **Validation & Metrics**

   - **Functional checks**:

     - Row routes ≤ k.
     - No duplicates per row.
     - Column totals ≤ original column sums.
     - Total routed bits ≥ total bits in S/T (`validate_phase_router`).

   - **Statistics**:

     - Row coverage histogram.
     - Column min/max/mean/std/skew.
     - Fill ratio = total active routes / (N × k).
     - Routes per row.

   - **Performance**:

     - Runtime for packing, routing, and total pipeline.

4. **Visual outputs (optional)**

   - Dump PBM/PNG for:

     - `S_rot`, `T_rot`, `S_shuf`, `T_shuf`, `T_final`, `O`.

   - Verify phase separation and uniform dispersal.

5. **Return**

   - Dictionary with metrics and validation status.

---

### **3.3 Outputs**

- `metrics` dict containing:

  - `N`, `k`, `active_routes`, `routes_per_row`, `fill_ratio`.
  - Column stats: `min`, `max`, `mean`, `std`, `skew`.
  - Runtime: `packing_time_ms`, `routing_time_ms`, `total_time_ms`.

- Optional PBM/PNG images.
- Validation flag (pass/fail).

---

## **4. `phase_router_run.py` Specification**

### **4.1 Inputs / Configuration**

- List of matrix sizes: `N_list = [256, 512, 1024, 2048, 4096]`.
- List of max routes: `k_list = [8, 16, 64, 256, 512, N]`.
- Number of OpenMP threads.
- Number of trials per configuration.
- Optional seed management: fixed vs random.
- Dump directory for PBM/PNG output.

---

### **4.2 Workflow**

1. **Loop over configurations**

   ```python
   for N in N_list:
       for k in k_list:
           for trial in range(num_trials):
               # optionally set seed
               # call phase_router_test.run_single_test(N, k, seed)
               # collect metrics
   ```

2. **Metric collection**

   - Combine metrics from multiple trials:

     - Mean ± std for active routes, fill ratio, column stats.
     - Runtime stats per stage.

   - Store in JSON or CSV for analysis.

3. **Scaling analysis**

   - Runtime vs N (log-log plot).
   - Runtime vs k.
   - OpenMP thread scaling: speedup vs thread count.

4. **Visual outputs**

   - Save representative PBM/PNG images for selected configurations.
   - Optional: small N (128–512) for phase visualization.

5. **Reproducibility**

   - Compare repeated runs with the same seed → identical outputs.
   - Compare random-seed runs → statistically consistent column distributions.

---

### **4.3 Outputs**

- **Metrics file** (JSON/CSV) per batch:

  - N, k, trial, active_routes, fill_ratio, routes_per_row, column stats, runtimes.

- **Visual outputs** (optional PBM/PNG).
- **Reproducibility logs**:

  - Seed, hash of routes array for regression.

---

## **5. `phase_router_vs_hash.py` Specification**

### **5.1 Overview**

This script performs **comprehensive stress tests** comparing the bit-packed phase router against a simple hash-based router. It includes single-phase routing tests and adversarial two-phase composability tests where hash routing typically collapses.

### **5.2 Inputs / Configuration**

- `N` – matrix size (default: 16,000)
- `ks_single` – list of k values for single-phase tests: `[16, 64, 256, 1024, 4096, 12800]`
- `ks_two` – list of k values for two-phase adversarial tests: `[16, 64, 256, 1024, 4096, 12800]`
- `--skip-plots` – optional flag to disable plotting for memory savings

### **5.3 Workflow**

1. **Single-Phase Stress Sweep**

   - Generate random binary matrices `S` and `T`
   - Run phase router and hash router on same inputs
   - Collect column load statistics and timing
   - Compare skew, fill ratios, and performance

2. **Two-Phase Adversarial Test**

   - **Phase 1**: Route with phase router and hash router
   - **Adversarial Construction**: Build worst-case `S2` matrix based on Phase 1 routes
   - **Phase 2**: Route again with both routers
   - Measure load collapse and composability failures

3. **Metric Collection**

   - Column statistics: min, max, mean, std, skew
   - Fill metrics: active routes, routes per row, fill ratio
   - Timing: phase1 + phase2 total runtime
   - System hardware metadata

### **5.4 Outputs**

- **CSV files**:

  - `single_phase_results.csv` – single-phase comparison metrics
  - `two_phase_adversarial_results.csv` – two-phase adversarial test results

- **Markdown table**:

  - `two_phase_adversarial_results.md` – human-readable summary with system specs

- **Optional plots** (in `test_output/plots/`):

  - Column load histograms comparing phase vs hash routers
  - Routing time vs k plots

- **System information** appended to markdown files:
  - OS, architecture, CPU model, cores, frequency, RAM

---

## **6. `phase_router_test_matrix.py` Specification**

### **6.1 Overview**

This script runs a **systematic test matrix** covering load balance, determinism, composability, and failure modes. It goes beyond single-run correctness to evaluate distributional behavior and worst-case scenarios.

### **6.2 Test Categories**

The test matrix includes 10 comprehensive test scenarios:

1. **Row-Degree Extremes** – Mixed sparse/dense rows
2. **Column-Target Stress** – Uneven column degrees
3. **Seed Reproducibility** – Deterministic behavior verification
4. **Monte Carlo Mean Load** – Seed-averaged statistical analysis
5. **Large-Scale Performance** – Scaling tests (N=1024, k=64)
6. **Adversarial Two-Phase** – Composability testing
7. **Edge-Case k Values** – Boundary conditions (k=1, k=N)
8. **Structured Sparse Patterns** – Diagonal/checkerboard patterns
9. **Phase Rotation Boundaries** – Wrap-around testing
10. **Hash Router Comparison** – Baseline comparison

### **6.3 Inputs / Configuration**

- `N` – matrix size (default: 1024)
- `k` – max connections per row (default: 64)
- `--skip-plots` – optional flag to disable plotting

### **6.4 Workflow**

1. **Test Execution**

   - Loop through all 10 test scenarios
   - Each test generates appropriate matrices and runs routing
   - Collect metrics and timing for each test

2. **Metric Collection**

   - Column load statistics: min, max, mean, std, skew
   - Runtime per test
   - System hardware metadata
   - Optional distribution plots

3. **Result Aggregation**

   - Combine results from all tests
   - Generate CSV and markdown outputs

### **6.5 Outputs**

- **CSV file**: `phase_router_test_matrix.csv` – machine-readable metrics
- **Markdown table**: `phase_router_test_matrix.md` – human-readable summary
- **Optional plots** (in `test_output/plots/`): Column load histograms
- **System information** appended to markdown files

---

## **7. Running the Test Suite**

### **7.1 Build Requirements**

```bash
# Build C++ extension first
python setup.py build_ext --inplace

# Install dependencies
pip install -r requirements.txt
```

### **7.2 Running Individual Tests**

```bash
# Single test
python evaluation/phase_router_test.py

# Batch scaling tests
python evaluation/phase_router_run.py

# Phase router vs hash comparison
python evaluation/phase_router_vs_hash.py --skip-plots

# Comprehensive test matrix
python evaluation/phase_router_test_matrix.py --skip-plots
```

### **7.3 Output Location**

All test outputs are saved to:

```text
test_output/
├── *.csv          # Machine-readable metrics
├── *.md           # Human-readable summaries
└── plots/         # Optional visual outputs
```

### **7.4 System Requirements**

- **Memory**: Large N/k values can consume significant memory
- **Dependencies**: matplotlib and psutil recommended for full functionality
- **Runtime**: Tests are designed to complete in reasonable time on modern hardware

---

## **8. Design Philosophy**

The test suite is designed to be:

- **Comprehensive**: Covering correctness, performance, and edge cases
- **Comparative**: Including baseline comparisons (hash router)
- **Reproducible**: With fixed seeds and system metadata
- **Scalable**: Supporting various matrix sizes and configurations
- **Analytical**: Providing both machine-readable and human-readable outputs

The tests answer critical questions about routing behavior under stress, composability, and adversarial conditions.
