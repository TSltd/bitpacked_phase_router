# NumPy example

import numpy as np
import router
import time

# -------------------- Configuration --------------------
N = 8192  # Number of rows/columns
k = 64    # Number of targets per source

# -------------------- Generate random binary matrices --------------------
print(f"Generating random binary matrices N={N}, k={k}...")
S = np.random.randint(0, 2, (N, N), dtype=np.uint8)
T = np.random.randint(0, 2, (N, N), dtype=np.uint8)

# Optional: precompute a column permutation for T
col_perm = np.arange(N) * 3 % N

# -------------------- Pack matrices --------------------
print("\nPacking matrices...")
t0 = time.time()
S_bits = router.pack_bits(S)
T_bits = router.pack_bits_T_permuted(T, col_perm)
t1 = time.time()
print(f"Packing time: {(t1 - t0)*1000:.2f} ms")

# -------------------- Allocate routes --------------------
routes = np.zeros((N, k), dtype=np.int32)

# -------------------- Run C++ router --------------------
print("\nRouting...")
t0 = time.time()
router.route_packed(S_bits, T_bits, np.arange(N, dtype=np.uint64), k, routes)
t1 = time.time()
print(f"C++ routing time: {(t1 - t0)*1000:.2f} ms")

# -------------------- Routing statistics --------------------
total_active = np.sum(routes != -1)
avg_per_row = total_active / N
fill_ratio = total_active / (N * k)

print("\n=== Routing stats ===")
print(f"Total active routes: {total_active}")
print(f"Average per row: {avg_per_row:.2f}")
print(f"Fill ratio: {fill_ratio:.3f}")
