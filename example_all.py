import numpy as np
import torch
import router
import time

# -------------------- Configuration --------------------
N = 8192   # rows/columns
k = 64     # targets per source

# -------------------- Generate structured binary matrices --------------------
print(f"Generating structured binary matrices N={N}, k={k}...")

# Source: left-aligned rows
S_np = np.zeros((N, N), dtype=np.uint8)
row_counts = np.random.randint(1, k+1, size=N)  # 1..k active entries per row
for i, count in enumerate(row_counts):
    S_np[i, :count] = 1

# Target: top-aligned columns
T_np = np.zeros((N, N), dtype=np.uint8)
col_counts = np.random.randint(1, k+1, size=N)  # 1..k active entries per column
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

# -------------------- 1️⃣ Raw NumPy matrices --------------------
print("\n=== Routing from raw NumPy matrices ===")
t0 = time.time()
router.router(S_np, T_np, k, routes_np)
t1 = time.time()
total_active = np.sum(routes_np != -1)
print(f"Routing time: {(t1 - t0)*1000:.2f} ms")
print(f"Total active routes: {total_active}")
print(f"Average per row: {total_active/N:.2f}")
print(f"Fill ratio: {total_active/(N*k):.3f}")

# -------------------- 2️⃣ PyTorch tensors --------------------
print("\n=== Routing from PyTorch tensors ===")
t0 = time.time()
router.router(S_torch, T_torch, k, routes_torch.numpy())
t1 = time.time()
total_active = (routes_torch != -1).sum().item()
print(f"Routing time: {(t1 - t0)*1000:.2f} ms")
print(f"Total active routes: {total_active}")
print(f"Average per row: {total_active/N:.2f}")
print(f"Fill ratio: {total_active/(N*k):.3f}")

# -------------------- 3️⃣ Pre-packed bit arrays --------------------
print("\n=== Routing from pre-packed bit arrays ===")
col_perm = np.arange(N) * 3 % N  # optional column permutation
t0 = time.time()
S_bits = router.pack_bits(S_np)
T_bits = router.pack_bits_T_permuted(T_np, col_perm)
t1 = time.time()
print(f"Packing time: {(t1 - t0)*1000:.2f} ms")

stats_bits = router.route_packed_with_stats(S_bits, T_bits, np.arange(N), k, routes_bits)
print(f"C++ routing time: {stats_bits['routing_time_ms']:.2f} ms")
print(f"Total active routes: {stats_bits['active_routes']}")
print(f"Average per row: {stats_bits['routes_per_row']:.2f}")
print(f"N: {stats_bits['N']}, k: {stats_bits['k']}")

# -------------------- 4️⃣ Pack-and-route (parallelized) --------------------
print("\n=== Pack-and-route from raw matrices (parallelized) ===")
stats_par = router.pack_and_route(S_np, T_np, k, routes_par)
print(f"Packing time (C++): {stats_par['packing_time_ms']:.2f} ms")
print(f"C++ routing time: {stats_par['routing_time_ms']:.2f} ms")
print(f"Total time: {stats_par['total_time_ms']:.2f} ms")
print(f"Total active routes: {stats_par['active_routes']}")
print(f"Average per row: {stats_par['routes_per_row']:.2f}")
print(f"N: {stats_par['N']}, k: {stats_par['k']}")
