## Why CPU (and not GPU)?

Although GPUs excel at dense numerical workloads, the **Bit-Packed Phase Router is intentionally designed for CPU execution**. Both the _algorithmic structure_ and the _target deployment environments_ favor CPUs over GPUs.

---

### 1. Many target systems are CPU-only

A core motivation for this project is routing in environments where:

- **No discrete GPU is available**
- Power, cost, or space constraints prohibit GPU use
- Deterministic behavior is required on commodity hardware

Examples include:

- Embedded systems and edge devices
- On-device neural routing or expert selection
- Simulation and analysis pipelines on CPU clusters
- Server-side systems where GPUs are reserved for training, not control logic

By running efficiently on CPUs, the router can be deployed **anywhere C++ runs**, without requiring specialized hardware.

---

### 2. Bitwise logic maps naturally to CPUs

Routing is dominated by:

- Bitwise AND operations
- Population count (`popcount`)
- Bit scans (`ctz`)
- Early termination once `k` routes are found

Modern CPUs execute these operations extremely efficiently:

- Single-cycle or near-single-cycle instructions
- Dedicated hardware support for `popcount`
- Strong branch prediction

GPUs, in contrast, are optimized for wide floating-point arithmetic and dense tensor operations, not irregular bitwise control flow.

---

### 3. Early termination causes GPU warp divergence

Each source row:

- Stops routing as soon as `k` valid targets are found
- Executes a variable amount of work depending on sparsity and alignment

This control flow is beneficial on CPUs but causes **warp divergence** on GPUs, forcing threads to serialize and reducing utilization.

The algorithm is intentionally structured to **avoid unnecessary work**, which aligns with CPU execution but conflicts with GPU execution models.

---

### 4. Cache-friendly access, not GPU-coalesced access

The router:

- Operates on compact bit-packed data
- Uses memory access patterns that fit in L1/L2 cache
- Minimizes memory traffic per routed connection

GPUs require:

- Coalesced global memory access
- Large, uniform memory streams

The router’s access pattern is cache-efficient on CPUs but **not well-suited to GPU memory hierarchies**.

---

### 5. Data transfer overhead outweighs GPU benefits

For realistic problem sizes:

- Bit-packed matrices and routing outputs can be tens of megabytes
- PCIe transfer latency can dominate total runtime

Since CPU routing already completes in milliseconds, **moving data to and from a GPU often costs more time than the routing itself**.

---

### 6. CPUs already achieve near-ideal performance

With bit-packing and OpenMP parallelism:

- Routing scales efficiently across CPU cores
- Performance is bounded by memory bandwidth and popcount throughput
- Modern CPUs approach the practical performance ceiling for this algorithm

In practice, adding a GPU would increase system complexity without meaningful speedup.

---

### 7. A GPU version would require a different algorithm

A GPU-friendly design would require:

- Fixed work per row (no early exits)
- Reduced control flow
- Dense tensor-style computation

These changes would undermine key guarantees:

- Deterministic phase separation
- Exact k-limited routing
- Bit-exact reproducibility

This project prioritizes **correctness, determinism, and deployability** over raw theoretical throughput.

---

### Summary

The Bit-Packed Phase Router is CPU-first by design:

- ✔ Runs efficiently on **CPU-only systems**
- ✔ Suitable for **embedded and edge deployments**
- ✔ Exploits fast CPU bitwise instructions and caches
- ✔ Avoids GPU-specific constraints and overhead

For this class of routing problem, **a well-optimized CPU implementation is both faster in practice and far more deployable than a GPU-based alternative**.

---
