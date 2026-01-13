## **Phase Router Testing & Evaluation Strategy**

### **1. Objectives**

The evaluation aims to:

1. Quantify **routing correctness**:

   - Ensure all active connections correspond to original S/T matrices.
   - Verify no duplicates, out-of-bounds indices, or missing routes.

2. Measure **performance metrics**:

   - Runtime for packing, routing, and combined operations.
   - Scaling with matrix size (N) and maximum connections (k).
   - Multi-threaded scaling (OpenMP).

3. Assess **load distribution and statistical properties**:

   - Row and column population distributions after routing.
   - Column load uniformity (min, max, mean, skew).
   - Fraction of active routes (`routes_per_row`, `fill ratio`).

4. Validate **visual and phase alignment**:

   - Intermediate PBM/PNG matrices.
   - Verify expected dispersal patterns and phase separation.

5. Ensure **reproducibility**:

   - Fixed seeds produce identical routing outputs.
   - Random seeds produce statistically consistent column distributions.

---

### **2. Test Types**

#### **2.1 Functional Correctness**

- **Unit tests for individual functions**:

  - `left_align_rows`: check row sums preserved, left-aligned ones.
  - `pack_bits` and unpack: round-trip equality.
  - `rotate_bits_full`: verify known patterns with offsets (including wraparound).
  - `permute_columns_bits`: known permutation → expected output.

- **Routing validation**:

  - Each row: routes ≤ `k`, no duplicates.
  - Each column: total routed connections ≤ original column sum.
  - Total routed bits ≥ total bits in S/T (as in `validate_phase_router`).

- **C++ vs Python consistency**:

  - Compare results of raw NumPy input, PyTorch input, and packed C++ routing.
  - Compare C++ auto-aligned vs Python-preprocessed alignment.

---

#### **2.2 Statistical Evaluation**

- **Column load distribution**:

  - Compute min, max, mean, standard deviation, and skew.
  - Compare against Poisson distribution for uniform random routing.
  - Compute "load balance ratio": `max / mean`.

- **Row coverage**:

  - Fraction of active routes per row: `routes_per_row`.
  - Histogram of row coverage (0…k).

- **Aggregate metrics**:

  - Fill ratio = total active routes / (N × k).
  - Fraction of rows achieving k routes.

---

#### **2.3 Performance Evaluation**

- Measure runtime for different stages:

  1. Row alignment (`left_align_rows`)
  2. Bit-packing (`pack_bits`)
  3. Routing (`phase_router_bitpacked` / `pack_and_route`)
  4. Total pipeline (packing + routing)

- Test scaling with varying:

  - Matrix sizes: N = 256, 512, 1024, 2048, 4096, 8192
  - Max connections: k = 8, 16, 64, 256, 512, N
  - Number of threads (`OMP_NUM_THREADS`)

- Visualize scaling:

  - Runtime vs N (log-log plot)
  - Runtime vs k
  - Speedup vs thread count

---

#### **2.4 Visual Evaluation**

- PBM → PNG conversion of intermediate matrices:

  - `S_rot`, `T_rot`, `S_phase`, `T_phase`, `S_shuf`, `T_shuf`, `T_final`, `O`

- Verify:

  - Phase separation (row-wise alignment)
  - Uniform dispersal
  - No missing or overlapping routes

- Include representative images in the paper (small N: 128–512)

---

#### **2.5 Reproducibility and Regression**

- **Fixed-seed runs**: identical outputs for multiple runs.
- **Random-seed runs**: statistics (mean/variance of column loads) remain consistent.
- Track active routes and column stats for regression testing.

---

### **3. Experimental Design**

1. **Matrix Generation**

   - Random binary matrices `S` and `T` with row sums drawn from uniform `[1, k_max]`.
   - Multiple sizes and densities to probe performance and statistical limits.

2. **Routing Variants**

   - Raw NumPy matrices → `router.router()`
   - PyTorch tensors → `router.router()`
   - Pre-packed bit arrays → `route_packed_with_stats()`
   - Pack-and-route (C++ auto-alignment) → `pack_and_route()`
   - Optional Python-aligned preprocessing → `pack_and_route()`

3. **Metrics Collected**

   - Active routes
   - Routes per row
   - Fill ratio
   - Column load statistics (min, max, mean, std, skew)
   - Runtime (ms) for packing, routing, total
   - Memory usage (optional)
   - PBM/PNG visual outputs

4. **Scaling Studies**

   - N ∈ {256, 512, 1024, 2048, 4096}
   - k ∈ {8, 16, 64, 256, 512, N}
   - Threads ∈ {1, 2, 4, 8, 16} for OpenMP

5. **Repeatability**

   - For each configuration, run 3–5 independent trials.
   - Report mean ± standard deviation for performance metrics and statistical measures.

---

### **4. Performance & Evaluation Section Outline (Paper)**

**4.1 Experimental Setup**

- Hardware description (CPU, cores, memory)
- Software (compiler flags, Python / PyTorch versions)
- Random seed management
- Matrix sizes and k ranges

**4.2 Correctness Validation**

- Row and column consistency
- Active route fraction
- Duplicate and out-of-bounds checks

**4.3 Statistical Properties**

- Column load distribution
- Row coverage histograms
- Fill ratio

**4.4 Performance Analysis**

- Runtime scaling with N, k
- OpenMP thread scaling (speedup)
- Pack-and-route vs pre-packed routing

**4.5 Visual Evaluation**

- Representative PBM/PNG images
- Phase separation, dispersal uniformity

**4.6 Discussion**

- Observed trends (e.g., column skew < 2× mean)
- Performance bottlenecks
- Reproducibility of results

---
