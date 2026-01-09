# Bit-Packed Phase Router

## Overview

The **Bit-Packed Phase Router** is a high-performance, memory-efficient C++ implementation (with Python bindings) for routing connections between sources and targets in large, dense matrices. It is designed to efficiently handle **dense or sparse connectivity patterns**, making it suitable for applications such as neural network simulations, mixture-of-experts (MoE) routing, and graph connectivity analysis.

The key feature of this router is **bit-level packing** of matrices combined with **row/column rotations and permutations**, allowing the simultaneous evaluation of multiple connections with minimal memory overhead.

---

## Features

- **Fully binary representation:** Each 64-bit word encodes 64 possible connections. Routing is done entirely with **bitwise operations**, maximizing speed and minimizing memory usage.
- **Deterministic phase rotations:** Rows and columns are rotated using precomputed offsets for spread-out routing.
- **Column pre-permutation:** Avoids runtime permutation overhead.
- **Deterministic load balancing:** Cumulative offsets for phase rotations ensure minimized concurrency.
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
3. **Compute Offsets:** Row and column rotations are determined based on the previous row/column bit sums. This maximizes phase separation, minimizing concurrency.
4. **Rotate & Permute:** Columns are pre-rotated to minimize runtime computation; source rows are rotated on-the-fly.
5. **Route:** For each source row, the router computes the bitwise AND with each target column, identifying active connections and filling the `routes` array up to `k` targets per source.

---

## Advantages

- **High performance:** Bitwise operations and column pre-rotation reduce computation time.
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

---

### 3️⃣ One-shot pack-and-route

```python
routes = np.zeros((N, k), dtype=np.int32)

stats = router.pack_and_route(S, T, k, routes)
print(f"Packing time: {stats['packing_time_ms']:.2f} ms")
print(f"C++ routing time: {stats['routing_time_ms']:.2f} ms")
print(f"Total time: {stats['total_time_ms']:.2f} ms")
```

- Accepts raw NumPy arrays.
- Packs matrices **in parallel** and calls the C++ router.
- Returns a stats dictionary including packing and routing times.

---

### 4️⃣ Helper functions

- `router.pack_bits(S)` – Pack a `uint8` matrix into bit-packed `uint64`.
- `router.pack_bits_T_permuted(T, col_perm)` – Pack with column permutation.
- `router.route_packed(S_bits, T_bits, row_perm, k, routes)` – Run routing on pre-packed arrays.
- `router.route_packed_with_stats(...)` – Same as above but returns stats.
- `router.pack_and_route(S, T, k, routes)` – One-shot packing + routing with timing stats.

---

## Performance Notes

- **Packing is the dominant cost** for large `N`.
- Routing itself is extremely fast: for `N=8192, k=64`:

| Method                        | Packing (ms) | C++ Routing (ms) | Total (ms) |
| ----------------------------- | ------------ | ---------------- | ---------- |
| Raw NumPy / PyTorch           | –            | ~1700            | ~1700      |
| Pre-packed arrays             | 2000–2200    | ~70              | 2070–2290  |
| Pack-and-route (parallelized) | 2000–2200    | ~70              | 2100–2300  |

- OpenMP parallelization provides a significant speed-up for packing and routing large matrices.

---

## Usage Example

### Structured matrices for effective routing

The Bit-Packed Phase Router is designed to **spread connections evenly** using row/column rotations and cumulative sums. For this to work properly, the matrices should **not be completely random**:

- **Source matrix (`S`)**: Rows are **left-aligned** — active entries (1s) start from the left of each row and the remaining positions are 0.
- **Target matrix (`T`)**: Columns are **top-aligned** — active entries (1s) start from the top of each column and the remaining positions are 0.

This structure ensures that the **phase rotations** (row/column offsets) actually separate connections across targets. Randomly distributed 1s would defeat this mechanism and make the router behave like a naive bitwise AND.

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
- Column pre-permutation and rotation ensure **deterministic and balanced routing**.
- Users can choose raw arrays, pre-packed arrays, or `pack_and_route` depending on workflow needs.

---
