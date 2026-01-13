# **Phase Router Python Test & Run Scripts – Specification**

## **1. Overview**

We have two scripts for testing the phase router:

- **`phase_router_test.py`** – executes **single tests** for a given matrix size (`N`) and routing degree (`k`). Performs correctness checks, statistics, and optional visual outputs.
- **`phase_router_run.py`** – executes **batches of tests** over multiple `N`, `k`, and threading configurations, collects metrics, and produces reproducibility and scaling data.

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
