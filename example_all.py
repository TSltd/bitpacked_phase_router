import numpy as np
import torch
import router
import time

def left_top_align(S, T):
    """
    Transform arbitrary binary matrices into left-aligned rows (S)
    and top-aligned columns (T) for proper phase-separated routing.
    """
    N = S.shape[0]

    # Left-align each row in S
    S_aligned = np.zeros_like(S)
    for i in range(N):
        ones_count = np.sum(S[i])
        S_aligned[i, :ones_count] = 1

    # Top-align each column in T
    T_aligned = np.zeros_like(T)
    for j in range(N):
        ones_count = np.sum(T[:, j])
        T_aligned[:ones_count, j] = 1

    return S_aligned, T_aligned

# -------------------- Configuration --------------------
N = 8192   # rows/columns
k = 64     # targets per source

# -------------------- Generate random binary matrices --------------------
print(f"Generating random binary matrices N={N}, k={k}...")

S_random = np.random.randint(0, 2, (N, N), dtype=np.uint8)
T_random = np.random.randint(0, 2, (N, N), dtype=np.uint8)

# Structured aligned matrices for comparison
print("Generating structured aligned matrices for comparison...")
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
routes_torch = torch.zeros((N, k), dtype=torch.int32)
routes_bits = np.zeros((N, k), dtype=np.int32)
routes_par = np.zeros((N, k), dtype=np.int32)

# -------------------- 1. Raw NumPy matrices --------------------
print("\n=== Routing from raw NumPy matrices ===")
t0 = time.time()
router.router(S_np, T_np, k, routes_np)
t1 = time.time()
total_active = np.sum(routes_np != -1)
print(f"Routing time: {(t1 - t0)*1000:.2f} ms")
print(f"Total active routes: {total_active}")
print(f"Average per row: {total_active/N:.2f}")
print(f"Fill ratio: {total_active/(N*k):.3f}")

# -------------------- 2. PyTorch tensors --------------------
print("\n=== Routing from PyTorch tensors ===")
t0 = time.time()
router.router(S_torch, T_torch, k, routes_torch.numpy())
t1 = time.time()
total_active = (routes_torch != -1).sum().item()
print(f"Routing time: {(t1 - t0)*1000:.2f} ms")
print(f"Total active routes: {total_active}")
print(f"Average per row: {total_active/N:.2f}")
print(f"Fill ratio: {total_active/(N*k):.3f}")

# -------------------- 3. Pre-packed bit arrays with random shuffling --------------------
print("\n=== Routing from pre-packed bit arrays with randomized permutations ===")
rng = np.random.default_rng(seed=42)  # optional seed
row_perm = np.arange(N)
col_perm = np.arange(N)
rng.shuffle(row_perm)
rng.shuffle(col_perm)

t0 = time.time()
S_bits = router.pack_bits(S_np)                       # rows in original order
T_bits = router.pack_bits_T_permuted(T_np, col_perm)  # permuted columns
t1 = time.time()
print(f"Packing time: {(t1 - t0)*1000:.2f} ms")

stats_bits = router.route_packed_with_stats(S_bits, T_bits, row_perm, k, routes_bits)
print(f"C++ routing time: {stats_bits['routing_time_ms']:.2f} ms")
print(f"Total active routes: {stats_bits['active_routes']}")
print(f"Average per row: {stats_bits['routes_per_row']:.2f}")
print(f"N: {stats_bits['N']}, k: {stats_bits['k']}")

# -------------------- 4. Pack-and-route (parallelized) --------------------
print("\n=== Pack-and-route from raw matrices (parallelized) ===")
stats_par = router.pack_and_route(S_np, T_np, k, routes_par)
print(f"Packing time (C++): {stats_par['packing_time_ms']:.2f} ms")
print(f"C++ routing time: {stats_par['routing_time_ms']:.2f} ms")
print(f"Total time: {stats_par['total_time_ms']:.2f} ms")
print(f"Total active routes: {stats_par['active_routes']}")
print(f"Average per row: {stats_par['routes_per_row']:.2f}")
print(f"N: {stats_par['N']}, k: {stats_par['k']}")

# -------------------- 5. Python preprocessing + pack-and-route --------------------
print("\n=== Python alignment preprocessing + pack-and-route ===")
routes_py_align = np.zeros((N, k), dtype=np.int32)
t0 = time.time()
S_py_aligned, T_py_aligned = left_top_align(S_random, T_random)
t1 = time.time()
print(f"Python alignment time: {(t1 - t0)*1000:.2f} ms")
stats_py = router.pack_and_route(S_py_aligned, T_py_aligned, k, routes_py_align)
print(f"C++ packing time: {stats_py['packing_time_ms']:.2f} ms")
print(f"C++ routing time: {stats_py['routing_time_ms']:.2f} ms")
print(f"Total time (Python align + C++): {(t1 - t0)*1000 + stats_py['total_time_ms']:.2f} ms")
print(f"Active routes: {stats_py['active_routes']}")
print(f"Average per row: {stats_py['routes_per_row']:.2f}")

# -------------------- 6. Automatic C++ alignment in pack-and-route --------------------
print("\n=== Automatic C++ alignment in pack-and-route (random input) ===")
routes_cpp_auto = np.zeros((N, k), dtype=np.int32)
stats_cpp_auto = router.pack_and_route(S_random, T_random, k, routes_cpp_auto)
print(f"C++ alignment + packing time: {stats_cpp_auto['packing_time_ms']:.2f} ms")
print(f"C++ routing time: {stats_cpp_auto['routing_time_ms']:.2f} ms")
print(f"Total time: {stats_cpp_auto['total_time_ms']:.2f} ms")
print(f"Active routes: {stats_cpp_auto['active_routes']}")
print(f"Average per row: {stats_cpp_auto['routes_per_row']:.2f}")

# -------------------- 7. Test C++ alignment functions separately --------------------
print("\n=== Testing C++ alignment functions ===")
t0 = time.time()
S_cpp_aligned = router.left_align_rows(S_random)
T_cpp_aligned = router.top_align_columns(T_random)
t1 = time.time()
print(f"C++ alignment time: {(t1 - t0)*1000:.2f} ms")

# Verify alignment preserves counts
print("Verifying alignment preserves row/column sums...")
for i in range(min(5, N)):
    orig_sum = np.sum(S_random[i])
    align_sum = np.sum(S_cpp_aligned[i])
    if orig_sum != align_sum:
        print(f"Row {i} sum mismatch: {orig_sum} vs {align_sum}")

for j in range(min(5, N)):
    orig_sum = np.sum(T_random[:, j])
    align_sum = np.sum(T_cpp_aligned[:, j])
    if orig_sum != align_sum:
        print(f"Column {j} sum mismatch: {orig_sum} vs {align_sum}")

print("Alignment verification complete.")
