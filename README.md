# Bit-Packed Phase Router

## **Overview**

The **Bit-Packed Phase Router** is a **high-performance, memory-efficient C++ library** (with Python bindings) for routing connections between sources and targets in large binary matrices. It is designed to efficiently handle **dense or sparse connectivity patterns**, making it ideal for **neural network simulations, mixture-of-experts (MoE) routing, graph connectivity analysis, and constrained routing tasks**.

The key innovation lies in **phase separation through deterministic rotations combined with full bit-packing**. Each 64-bit word encodes 64 possible connections, allowing routing entirely with **bitwise operations** for minimal memory overhead and maximum speed. **Phase rotations with cumulative offsets** ensure deterministic, balanced routing with minimal concurrency, while **k-limited routing in a single pass** guarantees each source row connects to a fixed number of targets efficiently.

The router supports both raw binary matrices and pre-packed bit arrays. When raw matrices are provided, the pack_and_route interface automatically performs alignment, optional deterministic or randomized shuffling, packing, and routing entirely in C++ for maximum performance. When pre-packed arrays are used, permutations must be supplied explicitly by the caller, enabling full control over routing behavior.

The router is **CPU-first by design**, leveraging fast bitwise operations, cache-efficient memory access, and early termination, making it well-suited for CPU-only and embedded systems (see [Why CPU?](docs/why_cpu.md)).

Additional features include:

- **Automatic matrix alignment** for optimal phase separation (left-aligned rows, top-aligned columns).
- **Parallelized execution with OpenMP** for large matrices (`N ≥ 8192`).
- **Flexible Python API** supporting raw NumPy arrays, PyTorch tensors, pre-packed bit arrays, or a one-shot `pack_and_route`.

By combining **binary operations, deterministic balancing, and parallelized routing**, the Bit-Packed Phase Router delivers **extremely fast, scalable, and reproducible routing**, outperforming conventional approaches in speed and memory efficiency.

---

## Features

- **Fully binary representation:** Each 64-bit word encodes 64 possible connections. Routing is done entirely with **bitwise operations**, maximizing speed and minimizing memory usage.
- **Automatic matrix alignment:** Accepts arbitrary binary matrices and automatically aligns them (left-aligned rows, top-aligned columns) for optimal phase separation.
- **Deterministic phase rotations:** Rows and columns are rotated using inline-computed offsets for spread-out routing.
- **Deterministic phase separation:** Cumulative bit-count–based offsets distribute routes evenly across phases.
- **Optional randomized or deterministic shuffling:** Improves saturation and reduces contention, performed inline in C++ or controlled by the user for pre-packed inputs.
- **k-limited routing:** Each source row can route to at most `k` active targets.
- **Parallelized with OpenMP:** Exploits multiple CPU threads for large matrices.
- **Memory-efficient:** Heap allocation for very large matrices (`N > 1024`).
- **Flexible Python API:** Supports raw NumPy arrays, PyTorch tensors, pre-packed arrays, or a one-shot `pack_and_route`.

---

## Novelty

The **Bit-Packed Phase Router** is novel because it combines:

1. **Fully binary bit-packed computation** — all routing is done with direct bitwise operations, which is faster and more memory-efficient than traditional integer or floating-point approaches.
2. **Phase rotations with cumulative offsets** — ensures deterministic and balanced routing across sources and targets, even in dense or clustered matrices.
3. **k-limited routing in a single pass** — guarantees a fixed number of targets per source without iterative searching.
4. **Parallelized, scalable design** — OpenMP allows efficient multi-threaded execution for very large networks.
5. **Flexible API for Python integration** — works with raw matrices, pre-packed bit arrays, PyTorch tensors, or combined packing + routing in one call.

This combination of **binary operations, deterministic balancing, and parallelized routing** is what sets this project apart from conventional routing or connectivity implementations.

---

## How It Works

1. **Populate:** Random or custom source (`S`) and target (`T`) matrices are generated and stored as bytes.
2. **Bit-Pack:** Matrices are converted to bit-packed form for high-performance bitwise operations.
3. **Compute Phase Offsets:**  
   Row and column rotation offsets are computed inline based on cumulative bit counts of preceding rows and columns. This deterministic process spreads connections across phases, minimizing contention.
4. **Apply Rotations:**  
   Source rows are rotated on-the-fly during routing. Target columns may be pre-permuted by the caller (for pre-packed inputs) or implicitly handled during packing for raw matrices.
5. **Route:** For each source row, the router computes the bitwise AND with each target column, identifying active connections and filling the `routes` array up to `k` targets per source.

---

## Advantages

- **High performance:** Bitwise operations and phase-separated rotations minimize computation time.
- **Deterministic load balancing:** Phase rotations minimize concurrency.
- **Scalable:** Can handle matrices up to `N = 8192+` with reasonable memory usage.
- **Parallel-friendly:** OpenMP allows multi-threaded execution for very large networks.
- **Flexible API:** Supports raw matrices, PyTorch tensors, pre-packed arrays, and a one-shot `pack_and_route`.

---

## Dependencies

- **Python 3.8+** (for Python bindings)
- **NumPy** and **PyTorch** (optional for tensor inputs)
- **C++ compiler** supporting C++11+
- **POSIX-compliant system** (for `clock_gettime`)
- **OpenMP** (optional but recommended for large `N`)
- **pybind11** and **numpy headers** for Python extension

---

## Compilation

```bash
# Build the Python extension in-place
python setup.py build_ext --inplace
```

This will generate a `router` module that can be imported in Python.

---

## Python API

### 1️⃣ Raw matrices

```python
import numpy as np
import router

N, k = 8192, 64
S = np.random.randint(0, 2, (N, N), dtype=np.uint8)
T = np.random.randint(0, 2, (N, N), dtype=np.uint8)
routes = np.zeros((N, k), dtype=np.int32)

router.router(S, T, k, routes)
```

- Input: raw NumPy array (`uint8`) or PyTorch tensor.
- Output: `routes` filled with up to `k` active targets per source.
- Unassigned routes are `-1`.

---

### 2️⃣ Pre-packed bit arrays

```python
S_bits = router.pack_bits(S)
T_bits = router.pack_bits_T_permuted(T, np.arange(N))
routes = np.zeros((N, k), dtype=np.int32)

stats = router.route_packed_with_stats(S_bits, T_bits, np.arange(N), k, routes)
print(stats['routing_time_ms'], stats['active_routes'])
```

- Input: bit-packed arrays (`uint64`).
- Optional column permutation to reduce runtime overhead.
- Returns detailed routing stats.

**Note:** When using pre-packed bit arrays, any desired row or column permutations must be supplied explicitly by the caller. No automatic shuffling is applied in this mode.

---

### 3️⃣ One-shot pack-and-route

```python
routes = np.zeros((N, k), dtype=np.int32)

stats = router.pack_and_route(S, T, k, routes)
print(f"Packing time: {stats['packing_time_ms']:.2f} ms")
print(f"C++ routing time: {stats['routing_time_ms']:.2f} ms")
print(f"Total time: {stats['total_time_ms']:.2f} ms")
```

- Accepts raw NumPy arrays (arbitrary alignment).
- Automatically aligns matrices for optimal phase separation.
- Packs matrices **in parallel** and calls the C++ router.
- Returns a stats dictionary including alignment, packing, and routing times.

This is the recommended entry point for most users. It automatically:

- Aligns arbitrary binary matrices
- Applies deterministic phase separation (and optional shuffling)
- Packs matrices in parallel
- Executes routing in C++

---

### 4️⃣ Helper functions

- `router.left_align_rows(S)` – Left-align rows of a `uint8` matrix.
- `router.top_align_columns(T)` – Top-align columns of a `uint8` matrix.
- `router.pack_bits(S)` – Pack a `uint8` matrix into bit-packed `uint64`.
- `router.pack_bits_T_permuted(T, col_perm)` – Pack with column permutation.
- `router.route_packed(S_bits, T_bits, row_perm, k, routes)` – Run routing on pre-packed arrays.
- `router.route_packed_with_stats(...)` – Same as above but returns stats.
- `router.pack_and_route(S, T, k, routes)` – One-shot packing + routing with automatic alignment and timing stats.

---

## Performance Notes

- **Packing dominates runtime** for large `N`. Routing itself is extremely fast once data are bit-packed.
- Overall complexity scales as **O(N²)** for packing/alignment and **O(N)** for routing.
- Automatic C++ alignment preserves full routing saturation while avoiding costly Python preprocessing.

The benchmarks below illustrate how different entry points trade off flexibility, preprocessing cost, and end-to-end performance.

---

### Large-scale benchmark: **N = 8192, k = 64**

| Method                                       | Packing / Align (ms) | C++ Routing (ms) | Total Time (ms) | Active Routes | Avg / Row |
| -------------------------------------------- | -------------------- | ---------------- | --------------- | ------------- | --------- |
| Raw NumPy matrices                           | –                    | 653.01           | 653.01          | 524,288       | 64.00     |
| PyTorch tensors                              | –                    | 486.09           | 486.09          | 524,288       | 64.00     |
| Pre-packed bit arrays (manual permutations)  | 405.08               | 64.89            | 469.97          | 524,288       | 64.00     |
| Pack-and-route (structured input)            | 513.11               | 51.12            | 564.53          | 524,288       | 64.00     |
| Python alignment + pack-and-route            | 2286.25              | 569.04\*         | 2855.60         | 524,288       | 64.00     |
| **Automatic C++ alignment + pack-and-route** | **731.05**           | **62.67**        | **794.11**      | **524,288**   | **64.00** |

- Includes Python alignment + C++ packing + routing.

**Observations**

- Once packed, routing completes in **~50–65 ms**, even at `N = 8192`.
- Python-side preprocessing dominates runtime and should be avoided for large matrices.
- Automatic C++ alignment is **~3.6× faster** than Python alignment while achieving identical saturation.

---

### Small-scale benchmark: **N = 256, k = 64**

| Method                                       | Total Time (ms) | Active Routes | Avg / Row |
| -------------------------------------------- | --------------- | ------------- | --------- |
| Raw NumPy matrices                           | ~2.2            | 16,352        | 63.88     |
| PyTorch tensors                              | ~6.8            | 16,352        | 63.88     |
| Pre-packed bit arrays                        | ~0.6            | 16,364        | 63.92     |
| Pack-and-route (structured input)            | ~0.40           | 16,352        | 63.88     |
| Python alignment + pack-and-route            | ~10.4           | 16,384        | 64.00     |
| **Automatic C++ alignment + pack-and-route** | **~0.31**       | **16,384**    | **64.00** |

**Observations**

- At small `N`, overhead dominates, but trends remain consistent.
- Automatic C++ alignment achieves **full saturation** while remaining over **30× faster** than Python preprocessing.
- Pre-packed inputs are fastest when permutations are managed explicitly by the user.

---

### Summary

- **Fastest routing:** Pre-packed bit arrays with user-controlled permutations
- **Best general-purpose API:** `pack_and_route` with automatic C++ alignment
- **Avoid for large N:** Python-side alignment or per-element Python routing

Automatic alignment and bit-packed routing allow the system to scale efficiently to large problem sizes while maintaining deterministic, fully saturated routing.

---

## Usage Example

### Matrix Alignment Options

The Bit-Packed Phase Router works optimally with **aligned matrices** for phase separation, but now **automatically handles arbitrary input matrices.**

#### Choosing the Right Entry Point

- Use `pack_and_route` if you have raw matrices and want maximum performance with minimal effort.
- Use `route_packed` / `route_packed_with_stats` only if you already manage bit-packing and permutations manually.

#### Option 1: Automatic Alignment (Recommended)

Simply pass any binary matrices to `pack_and_route` - alignment happens automatically:

```python
import numpy as np
import router

# Random or arbitrary matrices - no preprocessing needed!
S = np.random.randint(0, 2, (N, N), dtype=np.uint8)
T = np.random.randint(0, 2, (N, N), dtype=np.uint8)

routes = np.zeros((N, k), dtype=np.int32)
stats = router.pack_and_route(S, T, k, routes)  # Automatic alignment
```

#### Option 2: Manual Alignment (For Control)

Pre-align matrices manually using C++ or Python functions:

```python
# C++ alignment (fast)
S_aligned = router.left_align_rows(S)
T_aligned = router.top_align_columns(T)

# Python alignment (flexible)
from example_all import left_top_align
S_aligned, T_aligned = left_top_align(S, T)

# Then route normally
stats = router.pack_and_route(S_aligned, T_aligned, k, routes)
```

#### Option 3: Pre-structured Matrices (Optimal Performance)

For maximum performance, create aligned matrices directly:

- **Source matrix (`S`)**: **Left-aligned rows** — 1s at the start of each row
- **Target matrix (`T`)**: **Top-aligned columns** — 1s at the top of each column

This ensures the **phase rotations** separate connections optimally. Random distributions work but may reduce efficiency.

---

### Python example

```python
import numpy as np
import torch
import router
import time

N, k = 8192, 64

# -------------------- Structured matrices --------------------
S_np = np.zeros((N, N), dtype=np.uint8)
row_counts = np.random.randint(1, k+1, size=N)
for i, count in enumerate(row_counts):
    S_np[i, :count] = 1

T_np = np.zeros((N, N), dtype=np.uint8)
col_counts = np.random.randint(1, k+1, size=N)
for j, count in enumerate(col_counts):
    T_np[:count, j] = 1

# PyTorch versions
S_torch = torch.from_numpy(S_np)
T_torch = torch.from_numpy(T_np)

# Routing buffers
routes_np = np.zeros((N, k), dtype=np.int32)
routes_torch = np.zeros((N, k), dtype=np.int32)
routes_bits = np.zeros((N, k), dtype=np.int32)
routes_par = np.zeros((N, k), dtype=np.int32)

# -------------------- 1️⃣ Raw NumPy --------------------
router.router(S_np, T_np, k, routes_np)

# -------------------- 2️⃣ PyTorch tensors --------------------
router.router(S_torch, T_torch, k, routes_torch.numpy())

# -------------------- 3️⃣ Pre-packed bit arrays --------------------
col_perm = np.arange(N) * 3 % N
S_bits = router.pack_bits(S_np)
T_bits = router.pack_bits_T_permuted(T_np, col_perm)
router.route_packed_with_stats(S_bits, T_bits, np.arange(N), k, routes_bits)

# -------------------- 4️⃣ One-shot pack-and-route --------------------
router.pack_and_route(S_np, T_np, k, routes_par)
```

---

### Arbitrary Matrix Input and Automatic Alignment

The router now **automatically handles arbitrary binary matrices** without requiring manual preprocessing. If your matrices are not pre-aligned, the `pack_and_route` function will automatically:

1. **Left-align rows** in the source matrix (`S`)
2. **Top-align columns** in the target matrix (`T`)
3. Proceed with packing and routing

This ensures optimal phase separation regardless of input structure. For manual preprocessing or inspection, use the alignment functions directly:

```python
# Manual alignment
S_aligned = router.left_align_rows(S)
T_aligned = router.top_align_columns(T)

# Or use the Python function from example_all.py
from example_all import left_top_align
S_aligned, T_aligned = left_top_align(S, T)
```

**Example with random matrices:**

```python
import numpy as np
import router

N, k = 8192, 64

# Random matrices (no alignment needed)
S = np.random.randint(0, 2, (N, N), dtype=np.uint8)
T = np.random.randint(0, 2, (N, N), dtype=np.uint8)

routes = np.zeros((N, k), dtype=np.int32)
stats = router.pack_and_route(S, T, k, routes)

print(f"Total active routes: {stats['active_routes']}")
print(f"Average per row: {stats['routes_per_row']:.2f}")
```

The alignment preserves the total number of 1s per row/column, just reorganizes their positions for better routing performance.

---

### Key points

- Using **left-aligned rows** and **top-aligned columns** allows the **phase rotations and cumulative sum offsets** to evenly distribute routes.
- The router supports both **raw matrices** and **pre-packed bit arrays**, as well as **PyTorch tensors**.
- The **`k` parameter** limits the number of targets per source, and unused entries are filled with `-1`.
- For large `N`, **OpenMP parallelization** ensures efficient packing and routing.

---

## Applications

- Dense or sparse **neural network connectivity simulations**
- **Mixture-of-experts (MoE) routing**
- **Graph connectivity analysis**
- **Class pruning or constrained routing tasks**

---

## Notes

- Heap allocation is required for very large matrices (`N ≥ 1024`).
- Phase-separated rotations and optional permutations ensure deterministic and balanced routing.
- Users can choose raw arrays, pre-packed arrays, or `pack_and_route` depending on workflow needs.

---

## Reference

This implementation is based on the ideas introduced in:

[Phase-Separated Binary Coupling Matrices via Barrel-Shifting:
A Discrete Approach to Approximate Doubly Stochastic Transformations](docs/Phase-Separated_Binary_Coupling_Matrices.pdf)

D. Shirley, 2026

The codebase extends the original formulation with full bit-packing, deterministic phase rotations, and a production-ready C++ / Python implementation.
