import numpy as np
import torch
import router
import time
import os
from pathlib import Path

# Try to import PIL for PBM to PNG conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Note: PIL/Pillow not available - PBM to PNG conversion disabled")


def convert_pbm_to_png(pbm_files, invert=True, png_folder="dump/png"):
    png_folder = Path(png_folder)
    png_folder.mkdir(parents=True, exist_ok=True)
    png_files = []

    for pbm_file in pbm_files:
        try:
            im = Image.open(pbm_file).convert("L")
            if invert:
                im = Image.eval(im, lambda x: 255 - x)
            png_file = png_folder / (pbm_file.stem + ".png")
            im.save(png_file)
            png_files.append(png_file)
        except Exception as e:
            print(f"Failed to convert {pbm_file} to PNG: {e}")
    return png_files


def run_routing_with_pbm_stats(S, T, k, routes, phase_router_func, run_name):
    """Run routing with temporary PBM dump folder and convert PBMs to PNGs."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="pbm_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Run the routing function
        start_time = time.time()
        stats = phase_router_func(S, T, k, routes,
                                  dump=True,
                                  prefix=str(tmp_path))
        end_time = time.time()
        routing_time = (end_time - start_time) * 1000  # ms

        # Collect PBM files and convert to PNG
        pbm_files = sorted(tmp_path.glob("*.pbm"))
        png_folder = Path("dump/png") / run_name
        png_folder.mkdir(parents=True, exist_ok=True)
        png_files = convert_pbm_to_png(pbm_files, invert=True, png_folder=png_folder)

    print(f"PBM -> PNG conversion done for {len(png_files)} files")
    return routing_time, png_files


def left_top_align(S, T):
    """Left-align rows for phase separation (row-wise)."""
    N = S.shape[0]
    S_aligned = np.zeros_like(S)
    T_aligned = np.zeros_like(T)
    for i in range(N):
        S_aligned[i, :np.sum(S[i])] = 1
        T_aligned[i, :np.sum(T[i])] = 1
    return S_aligned, T_aligned


# -------------------- Configuration --------------------
N = 256  # rows/columns
k_max = 128  # maximum targets per row

# -------------------- Generate random binary matrices with independent row-wise counts --------------------
print(f"Generating random binary matrices N={N}, k_max={k_max}...")

rng_S = np.random.default_rng(seed=42)
rng_T = np.random.default_rng(seed=123)

row_counts_S = rng_S.integers(1, k_max + 1, size=N)
row_counts_T = rng_T.integers(1, k_max + 1, size=N)

S_random = np.zeros((N, N), dtype=np.uint8)
T_random = np.zeros((N, N), dtype=np.uint8)

for i in range(N):
    S_random[i, rng_S.choice(N, size=row_counts_S[i], replace=False)] = 1
    T_random[i, rng_T.choice(N, size=row_counts_T[i], replace=False)] = 1

# Make copies for legacy naming
S_np = S_random.copy()
T_np = T_random.copy()

# PyTorch versions
S_torch = torch.from_numpy(S_np)
T_torch = torch.from_numpy(T_np)

# -------------------- Routing buffers --------------------
routes_np = np.zeros((N, k_max), dtype=np.int32)
routes_torch = np.zeros((N, k_max), dtype=np.int32)
routes_bits = np.zeros((N, k_max), dtype=np.int32)
routes_par = np.zeros((N, k_max), dtype=np.int32)

# -------------------- 1. Raw NumPy matrices --------------------
print("\n=== Routing from raw NumPy matrices ===")
t0 = time.time()
router.router(S_np, T_np, k_max, routes_np)
t1 = time.time()
total_active = np.sum(routes_np != -1)
print(f"Routing time: {(t1 - t0)*1000:.2f} ms")
print(f"Total active routes: {total_active}")
print(f"Average per row: {total_active/N:.2f}")
print(f"Fill ratio: {total_active/(N*k_max):.3f}")

# -------------------- 2. PyTorch tensors --------------------
print("\n=== Routing from PyTorch tensors ===")
t0 = time.time()
router.router(S_torch, T_torch, k_max, routes_torch)
t1 = time.time()
total_active = np.sum(routes_torch != -1).item()
print(f"Routing time: {(t1 - t0)*1000:.2f} ms")
print(f"Total active routes: {total_active}")
print(f"Average per row: {total_active/N:.2f}")
print(f"Fill ratio: {total_active/(N*k_max):.3f}")

# -------------------- 3. Pre-packed bit arrays with independent shuffling --------------------
print("\n=== Routing from pre-packed bit arrays with independent shuffling ===")

rng = np.random.default_rng(42)

# permutations (ALL length N)
row_perm = np.arange(N, dtype=np.uint64)
col_perm_S = np.arange(N, dtype=np.uint64)
col_perm_T = np.arange(N, dtype=np.uint64)

rng.shuffle(row_perm)
rng.shuffle(col_perm_S)
rng.shuffle(col_perm_T)

# pack ORIGINAL matrices (no row permutation here!)
S_bits = router.pack_bits(S_random)
T_bits = router.pack_bits(T_random)

# choose k
k = k_max   # or any integer <= N

# route
stats_bits = router.route_packed_with_stats(
    S_bits,
    T_bits,
    row_perm,
    col_perm_S,
    col_perm_T,
    k,
    routes_bits
)


print(f"C++ routing time: {stats_bits['routing_time_ms']:.2f} ms")
print(f"Total active routes: {stats_bits['active_routes']}")
print(f"Average per row: {stats_bits['routes_per_row']:.2f}")
print(f"N: {stats_bits['N']}, k: {stats_bits['k']}")

# -------------------- 4. Pack-and-route (parallelized) --------------------
print("\n=== Pack-and-route from raw matrices (parallelized) ===")
routes_par = np.zeros((N, k_max), dtype=np.int32)
routing_time_par, png_files_par = run_routing_with_pbm_stats(
    S_random, T_random, k_max, routes_par,
    router.pack_and_route,
    run_name="pack_and_route"
)
print(f"Pack-and-Route routing time: {routing_time_par:.2f} ms")
print(f"PNG files: {len(png_files_par)}")

# -------------------- 5. Python preprocessing + pack-and-route --------------------
print("\n=== Python alignment preprocessing + pack-and-route ===")
routes_py_align = np.zeros((N, k_max), dtype=np.int32)
S_py_aligned, T_py_aligned = left_top_align(S_random, T_random)

routing_time_py, png_files_py = run_routing_with_pbm_stats(
    S_py_aligned, T_py_aligned, k_max, routes_py_align,
    router.pack_and_route,
    run_name="python_aligned"
)
print(f"Python-aligned routing time: {routing_time_py:.2f} ms")
print(f"PNG files: {len(png_files_py)}")

# -------------------- 6. Automatic C++ alignment in pack-and-route --------------------
print("\n=== Automatic C++ alignment in pack-and-route (random input) ===")
routes_cpp_auto = np.zeros((N, k_max), dtype=np.int32)
routing_time_cpp, png_files_cpp = run_routing_with_pbm_stats(
    S_random, T_random, k_max, routes_cpp_auto,
    router.pack_and_route,
    run_name="cpp_auto_aligned"
)
print(f"C++ auto-aligned routing time: {routing_time_cpp:.2f} ms")
print(f"PNG files: {len(png_files_cpp)}")

# -------------------- 7. Verify C++ alignment functions --------------------
print("\n=== Testing C++ alignment functions ===")
t0 = time.time()
S_cpp_aligned = router.left_align_rows(S_random)
T_cpp_aligned = router.left_align_rows(T_random)
t1 = time.time()
print(f"C++ alignment time: {(t1 - t0)*1000:.2f} ms")

# Check row sums
for i in range(5):
    if np.sum(S_random[i]) != np.sum(S_cpp_aligned[i]):
        print(f"Row {i} sum mismatch in S")
    if np.sum(T_random[i]) != np.sum(T_cpp_aligned[i]):
        print(f"Row {i} sum mismatch in T")

# -------------------- 8. PBM/PNG summary --------------------
png_files_groups = {
    "pack_and_route": png_files_par,
    "python_aligned": png_files_py,
    "cpp_auto_aligned": png_files_cpp,
}

for run_name, files in png_files_groups.items():
    print(f"\nRun: {run_name} ({len(files)} PNG files)")
    for png_file in sorted(files):
        size_mb = os.path.getsize(png_file) / (1024 * 1024)
        print(f"  • {png_file} ({size_mb:.2f} MB)")

print("\n=== All tests completed ===")
