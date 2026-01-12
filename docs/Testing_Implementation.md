# **Phase Router Testing & Evaluation Implementation Plan**

## **1. Goals**

- Implement a **robust testing suite** for the phase router (`router.cpp`) that:

  - Supports single-phase deterministic routing.
  - Measures correctness, statistics, performance, and reproducibility.
  - Generates intermediate visualizations for debugging.

- Replace the broken multiphase approach with a **functional, iterative testing framework** that still supports multiple N and k configurations.
- Maintain all existing C++ and Python API functionality:

  - `phase_router_bitpacked`
  - `pack_and_route`
  - `route_packed_with_stats`
  - `router`
  - PBM dumps and column statistics.

---

## **2. Core Modifications Needed**

### **2.1 Router C++ Layer**

- Keep the original `phase_router_bitpacked` and supporting functions **unchanged**.
- Add optional support for **iterative multi-k or staged routing** if desired:

  - A wrapper that can run `phase_router_bitpacked` multiple times over partially routed matrices.
  - Accumulate results while preserving **row and column constraints**.
  - Only if needed for testing multiple k values (not true multiphase random mixing).

### **2.2 Python Test Scripts**

- **phase_router_test.py**

  - Replace old multiphase calls with **single-phase calls** to:

    - `router.router()`
    - `pack_and_route()`
    - `route_packed_with_stats()`

  - Loop over different N and k configurations.
  - Perform the following checks for each run:

    - Row/column sums preserved.
    - Routes ≤ k per row.
    - Column loads ≤ original column sums.
    - No duplicates or out-of-bounds routes.
    - Validate total routed bits ≥ total bits in S/T.

  - Collect metrics:

    - Active routes, `routes_per_row`, fill ratio.
    - Column min/max/mean/skew.

- **phase_router_run.py**

  - Automate a **batch of experiments**:

    - Vary N ∈ {256, 512, 1024, 2048, 4096}
    - Vary k ∈ {8, 16, 64, 256, 512, N}
    - Optionally vary threads for OpenMP scaling.

  - Call `phase_router_test.py` functions or import the router API directly.
  - Save metrics and PBM/PNG outputs for visual inspection.
  - Maintain reproducibility:

    - Fixed seeds produce identical outputs.
    - Random seeds produce statistically consistent distributions.

---

## **3. Testing Plan**

### **3.1 Functional Correctness**

- **Unit tests**

  - `left_align_rows`: left-alignment check and row sum preservation.
  - `pack_bits` + unpack: round-trip equality.
  - `rotate_bits_full`: offset and wrap-around correctness.
  - `permute_columns_bits`: known permutation → expected output.

- **Routing validation**

  - Row routes ≤ k.
  - Column totals ≤ original column sums.
  - All bits routed as expected (use `validate_phase_router`).
  - No duplicates or out-of-bounds indices.

- **Consistency checks**

  - Compare results for:

    - Raw NumPy input → `router()`.
    - PyTorch tensors → `router()`.
    - Pre-packed arrays → `route_packed_with_stats()`.
    - Auto-aligned pack → `pack_and_route()`.

---

### **3.2 Statistical Evaluation**

- **Column statistics**

  - Compute min, max, mean, standard deviation, skew.
  - Compare against Poisson-like behavior.
  - Compute load balance ratio = max/mean.

- **Row coverage**

  - Fraction of active routes per row.
  - Histogram of coverage (0…k).

- **Aggregate metrics**

  - Fill ratio = total active routes / (N × k).
  - Fraction of rows achieving full k routes.

---

### **3.3 Performance Metrics**

- **Measure runtime**:

  1. Row alignment (`left_align_rows`).
  2. Bit-packing (`pack_bits`).
  3. Routing (`phase_router_bitpacked` / `pack_and_route`).
  4. Total pipeline.

- **Scaling studies**

  - Matrix size: N = 256 → 8192.
  - Max connections: k = 8 → N.
  - Threads: 1 → max available for OpenMP.

- **Plots**

  - Runtime vs N (log-log).
  - Runtime vs k.
  - OpenMP speedup vs thread count.

---

### **3.4 Visual Evaluation**

- **PBM → PNG conversion** for intermediate matrices:

  - `S_rot`, `T_rot`, `S_phase`, `T_phase`, `S_shuf`, `T_shuf`, `T_final`, `O`.

- **Verification**

  - Phase separation visible.
  - Uniform dispersal across columns.
  - No missing or overlapping routes.

- Representative images for N = 128–512.

---

### **3.5 Reproducibility**

- Fixed seed → identical outputs.
- Random seed → statistically consistent outputs.
- Store active route counts and column statistics for regression.

---

## **4. Implementation Steps**

1. **Preserve old working router**

   - Keep current `router.cpp` intact.
   - Ensure all original PBM dump and validation functions remain.

2. **Implement test harness**

   - `phase_router_test.py`

     - Functions for single N/k test.
     - Return full metrics dictionary + PBM dumps.

   - `phase_router_run.py`

     - Loop over multiple N, k, thread counts.
     - Collect all results, store in JSON/CSV.

3. **Replace broken multiphase**

   - Remove iterative multiphase logic in Python scripts.
   - Use loops over `k` or repeated single-phase routing if needed for testing.
   - Ensure `routes` array is cleared or accumulated correctly between runs.

4. **Add validation & metrics**

   - Compute row/column statistics.
   - Fill ratio, load balance, active routes per row.
   - Optional PBM/PNG dump for visual inspection.

5. **Run reproducibility tests**

   - Fixed seeds multiple runs → check equality.
   - Random seeds → check statistical distributions.

6. **Performance profiling**

   - Time each stage with `time.time()` or `now_ms()`.
   - Plot runtime scaling with N, k, threads.

7. **Document results**

   - Table of correctness metrics.
   - Histograms of column and row loads.
   - Representative PBM/PNG images.

---

## **5. Deliverables**

1. **Modified router scripts**

   - `phase_router_test.py` (single-phase, metrics collection).
   - `phase_router_run.py` (batch execution, scaling study).

2. **Metrics & visualizations**

   - JSON/CSV of all runs.
   - PBM/PNG of intermediate matrices.

3. **Validation outputs**

   - Column and row statistics.
   - Fill ratio and coverage histograms.

4. **Reproducibility logs**

   - Fixed-seed vs random-seed runs.

5. **Documentation**

   - Test plan Markdown (this document).
   - Notes on PBM/PNG inspection.
   - Summary of performance and correctness results.

---

This plan ensures:

- **All existing router functionality is preserved.**
- Broken multiphase logic is **safely replaced** with single-phase testing.
- The testing suite is **fully automated** for different N and k.
- Metrics, visuals, and reproducibility are systematically captured.

---
