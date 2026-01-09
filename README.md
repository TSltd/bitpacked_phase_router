# Bit-Packed Phase Router

## Overview

The **Bit-Packed Phase Router** is a high-performance, memory-efficient C implementation for routing connections between sources and targets in large, dense matrices. It is designed to efficiently handle **dense or sparse connectivity patterns**, making it suitable for applications such as neural network simulations, mixture-of-experts (MoE) routing, and graph connectivity analysis.

The key feature of this router is **bit-level packing** of matrices combined with **row/column rotations and permutations**, allowing the simultaneous evaluation of multiple connections with minimal memory overhead.

---

## Features

- **Bit-packed representation:** Each 64-bit word represents 64 possible connections.
- **Deterministic phase rotations:** Rows and columns are rotated using precomputed offsets for balanced, spread-out routing.
- **Column pre-permutation:** Avoids runtime permutation overhead.
- **k-limited routing:** Each source row can route to at most `k` active targets.
- **Parallelized with OpenMP:** Exploits multiple CPU threads for large matrices.
- **Memory-efficient:** Uses heap allocation for very large matrices (`N > 1024`).

---

## How It Works

1. **Populate:** Random or custom source (`S`) and target (`T`) matrices are generated and stored as bytes.
2. **Bit-Pack:** Matrices are converted to bit-packed form for high-performance bitwise operations.
3. **Compute Offsets:** Row and column rotations are determined based on the previous row/column bit sums, combined with a deterministic hash to spread routing evenly.
4. **Rotate & Permute:** Columns are pre-rotated to minimize runtime computation; source rows are rotated on-the-fly.
5. **Route:** For each source row, the router computes the bitwise AND with each target column, identifying active connections and filling the `routes` array up to `k` targets per source.

This order — **populate → pack → compute offsets → rotate → route** — ensures both correctness and maximum performance.

---

## Advantages

- **High performance:** Bitwise operations and column pre-rotation reduce computation time significantly.
- **Deterministic load balancing:** Phase rotations and hash-based offsets prevent hotspots in dense matrices.
- **Scalable:** Can handle matrices up to `N = 8192`+ with reasonable memory usage.
- **Parallel-friendly:** OpenMP allows multi-threaded execution for very large networks.
- **Flexible:** k-limited routing allows tuning of fan-out per source.

---

## Dependencies

- **C compiler** supporting C11 (or later)
- **POSIX-compliant system** (for `clock_gettime`)
- **OpenMP** for parallelization (optional but recommended for large `N`)

---

## Compilation

```bash
# Compile with optimization and OpenMP support
gcc -O2 -fopenmp -march=native router_test_heap.c -o router_test_heap
```

---

## Usage

```bash
# Run the router
./router_test_heap
```

**Example Output:**

```
Phase Router Bit-packed Test
----------------------------
N = 8192, k = 64
Routing time: 159.996348 ms
Total active routes: 524288
Average per source: 64.00
```

---

## Customization

- **Matrix size (`N`)**: Change `const size_t N = 8192;` in `main()`.
- **Maximum targets per source (`k`)**: Change `const size_t k = 64;`.
- **Number of threads**: Use `export OMP_NUM_THREADS=4` (or desired number).
- **Input matrices**: Replace random initialization with custom source and target matrices if needed.

---

## Applications

- Dense or sparse **neural network connectivity simulations**
- **Mixture-of-experts (MoE) routing**
- **Graph connectivity analysis**
- **Class pruning or constrained routing tasks**

---

## Notes

- Heap allocation is required for very large matrices; stack allocation is only suitable for `N ≤ 1024–2048`.
- Column pre-permutation and rotation ensure **deterministic and balanced routing**, which is critical for load balancing in large networks.

---
