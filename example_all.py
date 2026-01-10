import numpy as np
import torch
import router
import time
import os

# Try to import PIL for PBM to PNG conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Note: PIL/Pillow not available - PBM to PNG conversion disabled")

def convert_pbm_to_png(pbm_files, invert=True):
    png_files = []
    for pbm_file in pbm_files:
        try:
            im = Image.open(pbm_file)

            # Convert to grayscale ('L') so we can invert
            im = im.convert("L")

            if invert:
                # Invert black/white
                im = Image.eval(im, lambda x: 255 - x)

            png_file = os.path.splitext(pbm_file)[0] + ".png"
            im.save(png_file)
            png_files.append(png_file)
        except Exception as e:
            print(f"⚠️ Failed to convert {pbm_file} to PNG: {e}")
    return png_files

# Collect PBM files including the new routes dump
pbm_files_found = sorted([f for f in os.listdir(".") if f.endswith(".pbm")])

# Optional: filter for debug files (if you want only the ones from this run)
# pbm_files_found = [f for f in pbm_files_found if prefix in f]

# Convert to PNG format for easier viewing
print("\n🔄 Converting PBM files to PNG format...")
png_files = convert_pbm_to_png(pbm_files_found, invert=True)

if png_files:
    print(f"✅ Converted {len(png_files)} files to PNG format")
    print("PNG files created:")
    for png_file in sorted(png_files):
        size_mb = os.path.getsize(png_file) / (1024 * 1024)
        print(f"  • {png_file} ({size_mb:.2f} MB)")

    print("\n📷 You can now view the images with any standard image viewer:")
    print("  • feh *.png          (Linux)")
    print("  • eog *.png          (GNOME)")
    print("  • open *.png        (macOS)")
    print("  • start *.png       (Windows)")
    print("  • Any web browser or image viewer")
else:
    print("⚠️ No PNG files created (PIL not available or conversion failed)")
    print("\nTo view the PBM files directly, you can use:")
    print("  • feh *.pbm          (Linux image viewer)")
    print("  • display *.pbm      (ImageMagick)")
    print("  • gimp *.pbm        (GIMP)")

def left_top_align(S, T):
    """
    Transform binary matrices into:
    - S: left-aligned rows
    - T: left-aligned rows (instead of top-aligned columns)
    """
    N = S.shape[0]

    # Left-align each row in S (unchanged)
    S_aligned = np.zeros_like(S)
    for i in range(N):
        ones_count = np.sum(S[i])
        S_aligned[i, :ones_count] = 1

    # Left-align each row in T (row-oriented)
    T_aligned = np.zeros_like(T)
    for i in range(N):
        ones_count = np.sum(T[i])  # row sum, not column
        T_aligned[i, :ones_count] = 1

    return S_aligned, T_aligned



# -------------------- Configuration --------------------
N = 256   # rows/columns
k = 4     # targets per source

# -------------------- Generate random binary matrices --------------------
print(f"Generating random binary matrices N={N}, k={k}...")

# Random binary matrix with at most k ones per row
S_random = np.zeros((N, N), dtype=np.uint8)
T_random = np.zeros((N, N), dtype=np.uint8)

rng = np.random.default_rng(seed=42)  # optional seed for reproducibility

for i in range(N):
    ones_idx = rng.choice(N, size=k, replace=False)  # pick k unique columns
    S_random[i, ones_idx] = 1
    ones_idx = rng.choice(N, size=k, replace=False)
    T_random[i, ones_idx] = 1


# Structured aligned matrices for comparison
print("Generating structured aligned matrices for comparison...")
S_np = np.zeros((N, N), dtype=np.uint8)
row_counts = np.random.randint(1, k+1, size=N)
for i, count in enumerate(row_counts):
    S_np[i, :count] = 1

T_np = np.zeros((N, N), dtype=np.uint8)
row_counts = np.random.randint(1, k+1, size=N)
for i, count in enumerate(row_counts):
    T_np[i, :count] = 1

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

# -------------------- 3. Pre-packed bit arrays with random row-wise shuffling --------------------
print("\n=== Routing from pre-packed bit arrays with randomized row-wise permutations ===")
rng = np.random.default_rng(seed=42)  # optional seed
row_perm_S = np.arange(N)
row_perm_T = np.arange(N)
rng.shuffle(row_perm_S)  # shuffle rows of S
rng.shuffle(row_perm_T)  # shuffle rows of T

t0 = time.time()
S_bits = router.pack_bits(S_np)                  # pack S rows as usual
T_rowwise = T_np[row_perm_T, :]                  # apply row permutation to T
T_bits = router.pack_bits(T_rowwise)            # pack T rows now (row-oriented)
t1 = time.time()
print(f"Packing time: {(t1 - t0)*1000:.2f} ms")

stats_bits = router.route_packed_with_stats(S_bits, T_bits, row_perm_S, k, routes_bits)
print(f"C++ routing time: {stats_bits['routing_time_ms']:.2f} ms")
print(f"Total active routes: {stats_bits['active_routes']}")
print(f"Average per row: {stats_bits['routes_per_row']:.2f}")
print(f"N: {stats_bits['N']}, k: {stats_bits['k']}")


# -------------------- 4. Pack-and-route (parallelized) --------------------
print("\n=== Pack-and-route from raw matrices (parallelized) ===")

# Enable PBM dumping if matrix size is reasonable
dump_enabled = N <= 4096
dump_prefix = "test4_structured"

if dump_enabled:
    print(f"  🖼️  PBM dumping enabled (N={N} ≤ 4096)")
else:
    print(f"  🚫  Image dumping disabled - matrix too large (N={N} > 4096)")

stats_par = router.pack_and_route(S_np, T_np, k, routes_par,
                                dump=dump_enabled, prefix=dump_prefix)
print(f"Packing time (C++): {stats_par['packing_time_ms']:.2f} ms")
print(f"C++ routing time: {stats_par['routing_time_ms']:.2f} ms")
print(f"Total time: {stats_par['total_time_ms']:.2f} ms")
print(f"Total active routes: {stats_par['active_routes']}")
print(f"Average per row: {stats_par['routes_per_row']:.2f}")
print(f"N: {stats_par['N']}, k: {stats_par['k']}")

# Verify PBM files if dumping was enabled
if dump_enabled:
    import os
    expected_files = [
        f"{dump_prefix}_S.pbm",
        f"{dump_prefix}_T.pbm",
        f"{dump_prefix}_S_bits.pbm",
        f"{dump_prefix}_T_bits.pbm",
        f"{dump_prefix}_routes.pbm"
    ]
    files_created = sum(1 for f in expected_files if os.path.exists(f))
    print(f"  📁 PBM files created: {files_created}/{len(expected_files)}")

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

# Enable PBM dumping if matrix size is reasonable
dump_enabled = N <= 4096
dump_prefix = "test6_random"

if dump_enabled:
    print(f"  🖼️  PBM dumping enabled (N={N} ≤ 4096)")
else:
    print(f"  🚫  Image dumping disabled - matrix too large (N={N} > 4096)")

routes_cpp_auto = np.zeros((N, k), dtype=np.int32)
stats_cpp_auto = router.pack_and_route(S_random, T_random, k, routes_cpp_auto,
                                     dump=dump_enabled, prefix=dump_prefix)
print(f"C++ alignment + packing time: {stats_cpp_auto['packing_time_ms']:.2f} ms")
print(f"C++ routing time: {stats_cpp_auto['routing_time_ms']:.2f} ms")
print(f"Total time: {stats_cpp_auto['total_time_ms']:.2f} ms")
print(f"Active routes: {stats_cpp_auto['active_routes']}")
print(f"Average per row: {stats_cpp_auto['routes_per_row']:.2f}")

# Verify PBM files if dumping was enabled
if dump_enabled:
    import os
    expected_files = [
        f"{dump_prefix}_S.pbm",
        f"{dump_prefix}_T.pbm",
        f"{dump_prefix}_S_bits.pbm",
        f"{dump_prefix}_T_bits.pbm",
        f"{dump_prefix}_O.pbm"
    ]
    files_created = sum(1 for f in expected_files if os.path.exists(f))
    print(f"  📁 PBM files created: {files_created}/{len(expected_files)}")

# -------------------- 7. Test C++ alignment functions separately --------------------
print("\n=== Testing C++ alignment functions ===")
t0 = time.time()
S_cpp_aligned = router.left_align_rows(S_random)
T_cpp_aligned = router.left_align_rows(T_random)  # <-- row-aligned now
t1 = time.time()
print(f"C++ alignment time: {(t1 - t0)*1000:.2f} ms")

# Verify alignment preserves counts
print("Verifying alignment preserves row sums...")
for i in range(min(5, N)):
    orig_sum_S = np.sum(S_random[i])
    align_sum_S = np.sum(S_cpp_aligned[i])
    if orig_sum_S != align_sum_S:
        print(f"Row {i} sum mismatch in S: {orig_sum_S} vs {align_sum_S}")

    orig_sum_T = np.sum(T_random[i])
    align_sum_T = np.sum(T_cpp_aligned[i])
    if orig_sum_T != align_sum_T:
        print(f"Row {i} sum mismatch in T: {orig_sum_T} vs {align_sum_T}")

print("Alignment verification complete.")

# -------------------- 8. PBM Dumping Summary --------------------
print("\n=== PBM Dumping Summary ===")
import os

# Check for all possible PBM files that might have been created
possible_files = [
    "test4_structured_S.pbm", "test4_structured_T.pbm", "test4_structured_S_bits.pbm",
    "test4_structured_T_bits.pbm", "test4_structured_S_rot.pbm", "test4_structured_T_rot.pbm",
    "test4_structured_S_shuf.pbm", "test4_structured_T_shuf.pbm", "test4_structured_O.pbm",
    "test6_random_S.pbm", "test6_random_T.pbm", "test6_random_S_bits.pbm",
    "test6_random_T_bits.pbm", "test6_random_S_rot.pbm", "test6_random_T_rot.pbm",
    "test6_random_S_shuf.pbm", "test6_random_T_shuf.pbm", "test6_random_O.pbm"
]

# Collect PBM files including the new routes dump
pbm_files_found = sorted([f for f in os.listdir(".") if f.endswith(".pbm")])

# Convert PBM files to PNG using the function
print("\n🔄 Converting PBM files to PNG format...")
png_files = convert_pbm_to_png(pbm_files_found, invert=True)  # <-- call the function here

if png_files:
    print(f"✅ Converted {len(png_files)} files to PNG format")
    for png_file in sorted(png_files):
        size_mb = os.path.getsize(png_file) / (1024 * 1024)
        print(f"  • {png_file} ({size_mb:.2f} MB)")


print("\n=== All tests completed ===")
