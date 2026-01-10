#PBM dumping

1. **Confirm what representations you currently have** (important for correctness)
2. Give you **drop-in PBM writers** for:

   - raw `uint8_t` matrices (your S/T / aligned stages)
   - bit-packed `uint64_t` matrices (your internal representation)

3. Show **where to hook them** safely so they don’t interfere with OpenMP or routing

---

## 1. What you already have (good news)

You already maintain **three clean stages** that map naturally to images:

| Stage     | Representation                       | Perfect PBM source |
| --------- | ------------------------------------ | ------------------ |
| Raw input | `uint8_t S_np`, `T_np`               | dump directly      |
| Aligned   | `S_aligned`, `T_aligned`             | dump directly      |
| Packed    | `uint64_t S_bits`, `T_bits`, `T_rot` | unpack-on-write    |

Also important:

- Your **bit meaning matches PBM**:

  - `1 = black`
  - `0 = white`

- Your bit packing uses **LSB = column 0**, which is fine as long as we mirror that on output.

---

## 2. PBM writer for `uint8_t` matrices (simplest)

Use this for:

- `S_np`
- `T_np`
- `S_aligned`
- `T_aligned`

### Minimal, fast, safe

```cpp
#include <cstdio>
#include <cstdint>

void dump_pbm_u8(
    const char* filename,
    const uint8_t* M,
    size_t N
) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    fprintf(f, "P4\n%zu %zu\n", N, N);

    size_t row_bytes = (N + 7) >> 3;
    std::vector<uint8_t> row(row_bytes);

    for (size_t i = 0; i < N; i++) {
        std::fill(row.begin(), row.end(), 0);
        for (size_t j = 0; j < N; j++) {
            if (M[i * N + j]) {
                row[j >> 3] |= 0x80 >> (j & 7);
            }
        }
        fwrite(row.data(), 1, row_bytes, f);
    }

    fclose(f);
}
```

### Example usage

```cpp
dump_pbm_u8("S_input.pbm", S_np.data(), N);
dump_pbm_u8("S_aligned.pbm", (uint8_t*)S_aligned.data(), N);
```

---

## 3. PBM writer for **bit-packed matrices** (`uint64_t`)

This is the one you really want for debugging the router internals.

### Important mapping

Your packing:

```cpp
word |= 1ULL << b;  // b = column within word
```

PBM expects:

- **MSB first per byte**

So we must reverse bit order per byte on write.

### Fast unpack-on-write PBM dump

```cpp
void dump_pbm_bits(
    const char* filename,
    const uint64_t* bits,
    size_t N,
    size_t NB_words
) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    fprintf(f, "P4\n%zu %zu\n", N, N);

    size_t row_bytes = (N + 7) >> 3;
    std::vector<uint8_t> row(row_bytes);

    for (size_t i = 0; i < N; i++) {
        std::fill(row.begin(), row.end(), 0);

        const uint64_t* src = &bits[i * NB_words];
        for (size_t j = 0; j < N; j++) {
            size_t w = j >> 6;
            size_t b = j & 63;
            if (src[w] & (1ULL << b)) {
                row[j >> 3] |= 0x80 >> (j & 7);
            }
        }

        fwrite(row.data(), 1, row_bytes, f);
    }

    fclose(f);
}
```

---

## 4. Where to hook this in _your_ code

### 🔍 After alignment (most useful early check)

Inside `pack_and_route`:

```cpp
dump_pbm_u8("S_aligned.pbm", S_data, N);
dump_pbm_u8("T_aligned.pbm", T_data, N);
```

You’ll immediately see:

- rows left-aligned
- columns top-aligned

---

### 🔍 After packing

Right after packing loops:

```cpp
dump_pbm_bits("S_bits.pbm", S_bits.data(), N, NB_words);
dump_pbm_bits("T_bits.pbm", T_bits.data(), N, NB_words);
```

This confirms:

- no bit inversions
- no transpose mistakes
- correct row/column semantics

---

### 🔍 After column rotation (deep kernel debugging)

Inside `phase_router_bitpacked`, right after Step 2:

```cpp
dump_pbm_bits("T_rot.pbm", T_rot.data(), N, NB_words);
```

This is _extremely_ valuable for validating:

- rotation offsets
- permutation correctness
- wraparound behavior

---

## 5. Threading & performance notes

- **Never call dump functions inside `#pragma omp parallel`**
- Dump once per stage, not per row
- For large N, gate with a debug flag:

```cpp
if (N <= 2048) dump_pbm_bits(...);
```

PBM writing is still **orders of magnitude faster than PNG**, and zero dependencies.

---

## 6. How to view

```bash
display S_aligned.pbm
feh T_rot.pbm
gimp S_bits.pbm
```

Black pixels = `1`
White pixels = `0`

---

So the task is to add:

- `dump_pbm_u8()` for raw/aligned matrices
- `dump_pbm_bits()` for packed stages

Below is a concrete, low-risk plan that fits your codebase and workflow.

---

## Goals restated (to anchor the design)

You want to optionally dump:

1. **S** (input or aligned, depending on entry point)
2. **T**
3. **O** = routed paths matrix (derived from `routes`)

Constraints:

- **No impact on hot loops**
- **No PBM I/O during OpenMP regions**
- **No Python-side I/O**
- **Toggling must be explicit and cheap**
- Files written **once per call**, after routing

This is exactly the right mindset.

---

## High-level design (recommended)

### ✅ Use a **compile-time flag + runtime toggle**

This gives you:

- **Zero cost** when compiled out
- Ability to enable dumping only when you want

We’ll do **both**, layered:

```cpp
// Compile-time
#define ROUTER_ENABLE_DUMP 1   // set to 0 for production builds
```

```cpp
// Runtime (fast branch)
struct DumpConfig {
    bool enabled;
    const char* prefix;  // e.g. "run1"
};
```

When `enabled == false`, cost is literally one `if`.

---

## Where dumping should live (important)

**Only in top-level entry points**, never in kernels:

Good places:

- `router(...)`
- `pack_and_route(...)`
- `route_packed_with_stats(...)`

Bad places:

- `phase_router_bitpacked`
- inside `#pragma omp parallel`

We dump **after routing completes**, using already-available buffers.

---

## What exactly to dump

### 1️⃣ S and T

Depending on entry point:

| Entry point                 | Dump S                               | Dump T |
| --------------------------- | ------------------------------------ | ------ |
| `router()`                  | raw `S_np`, `T_np`                   | yes    |
| `pack_and_route()`          | **aligned** `S_aligned`, `T_aligned` | yes    |
| `route_packed_with_stats()` | optional (skip or unpack bits)       |        |

This matches how _you_ reason about correctness.

---

### 2️⃣ O (routed paths matrix)

Your `routes` is shape `(N, k)` where each entry is a **column index**.

We convert this to an **NxN binary image**:

```text
O[i, j] = 1  if column j was routed for row i
        = 0  otherwise
```

Multiple routes per row → multiple black pixels.

---

## PBM writers (recap, minimal)

You already saw these, but now we’ll **wire them cleanly**.

### `uint8_t` matrix → PBM

```cpp
void dump_pbm_u8(const char* filename,
                 const uint8_t* M,
                 size_t N);
```

### `routes` → PBM (O matrix)

```cpp
void dump_pbm_routes(const char* filename,
                     const int* routes,
                     size_t N,
                     size_t k)
{
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    fprintf(f, "P4\n%zu %zu\n", N, N);
    size_t row_bytes = (N + 7) >> 3;
    std::vector<uint8_t> row(row_bytes);

    for (size_t i = 0; i < N; i++) {
        std::fill(row.begin(), row.end(), 0);

        for (size_t r = 0; r < k; r++) {
            int j = routes[i * k + r];
            if (j >= 0) {
                row[j >> 3] |= 0x80 >> (j & 7);
            }
        }

        fwrite(row.data(), 1, row_bytes, f);
    }

    fclose(f);
}
```

This is **O(active_routes)**, not O(N²).

---

## Wiring this into your C++ (example: `pack_and_route`)

Add an **optional argument** from Python:

```cpp
py::dict pack_and_route(py::array_t<uint8_t> S_np,
                        py::array_t<uint8_t> T_np,
                        size_t k,
                        py::array_t<int> routes_np,
                        bool dump = false,
                        const std::string& prefix = "dump")
```

Bind it with defaults so existing code is untouched.

---

### Inside `pack_and_route` (near the end)

```cpp
#if ROUTER_ENABLE_DUMP
if (dump) {
    dump_pbm_u8((prefix + "_S.pbm").c_str(),
                (uint8_t*)S_aligned.data(), N);

    dump_pbm_u8((prefix + "_T.pbm").c_str(),
                (uint8_t*)T_aligned.data(), N);

    dump_pbm_routes((prefix + "_O.pbm").c_str(),
                    routes, N, k);
}
#endif
```

That’s it.

No timing distortion:

- One `if`
- Sequential I/O
- After routing completes
- Outside OpenMP

---

## Python-side usage (clean & explicit)

Add one debug run:

```python
stats = router.pack_and_route(
    S_np, T_np, k, routes_par,
    dump=True,
    prefix="debug_run"
)
```

Files produced:

```
debug_run_S.pbm
debug_run_T.pbm
debug_run_O.pbm
```

Open them after:

```bash
feh debug_run_*.pbm
```

---

## Performance impact (realistically)

| Scenario             | Impact                  |
| -------------------- | ----------------------- |
| Dump disabled        | **Zero measurable**     |
| Dump enabled (8192²) | dominated by disk write |
| Routing kernel       | untouched               |
| OpenMP behavior      | unchanged               |

You can also guard dumping with:

```cpp
if (dump && N <= 4096)
```

for sanity.

---
