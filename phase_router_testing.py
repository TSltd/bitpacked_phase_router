"""
phase_router_testing.py

Structured testing and evaluation protocol for the bit-packed phase router.
Collects:
- Correctness metrics (row/column coverage)
- Statistical metrics (load balance)
- Performance metrics (timing)
- Visual outputs (PBM -> PNG)
- Scaling and reproducibility studies
"""

import numpy as np
import torch
import router
import time
from pathlib import Path
import os

# Optional: PIL for PNG conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available: PBM -> PNG conversion disabled.")


# -----------------------------
# Utility functions
# -----------------------------

def convert_pbm_to_png(pbm_files, invert=True, png_folder="dump/png"):
    png_folder = Path(png_folder)
    png_folder.mkdir(parents=True, exist_ok=True)
    png_files = []
    if not PIL_AVAILABLE:
        return png_files

    for pbm_file in pbm_files:
        try:
            im = Image.open(pbm_file).convert("L")
            if invert:
                im = Image.eval(im, lambda x: 255 - x)
            png_file = png_folder / (pbm_file.stem + ".png")
            im.save(png_file)
            png_files.append(png_file)
        except Exception as e:
            print(f"Failed to convert {pbm_file}: {e}")
    return png_files


def collect_metrics(routes, S, T, k):
    """
    Compute correctness and statistics.
    routes: N x k int array
    S, T: N x N binary matrices
    k: max routes per row
    Returns dict of metrics
    """
    N = S.shape[0]
    active_routes = np.sum(routes != -1)
    routes_per_row = np.sum(routes != -1, axis=1)
    avg_routes_per_row = np.mean(routes_per_row)
    fill_ratio = active_routes / (N * k)

    # Column coverage
    O_bits = np.zeros_like(S)
    for i in range(N):
        for r in range(k):
            j = routes[i, r]
            if j >= 0:
                O_bits[i, j] = 1
    col_counts = np.sum(O_bits, axis=0)
    col_min = np.min(col_counts)
    col_max = np.max(col_counts)
    col_mean = np.mean(col_counts)
    col_std = np.std(col_counts)
    col_skew = col_max / (col_mean + 1e-9)

    # Check coverage vs original matrices
    total_bits_S = np.sum(S)
    total_bits_T = np.sum(T)
    coverage_ok = total_bits_S <= active_routes and total_bits_T <= active_routes

    metrics = {
        "active_routes": active_routes,
        "routes_per_row_mean": avg_routes_per_row,
        "fill_ratio": fill_ratio,
        "col_min": col_min,
        "col_max": col_max,
        "col_mean": col_mean,
        "col_std": col_std,
        "col_skew": col_skew,
        "coverage_ok": coverage_ok
    }
    return metrics


def run_routing_experiment(S, T, k, phase_router_func, dump_prefix=None, validate=True):
    """
    Run a routing experiment.
    Returns:
      metrics dict, routing_time_ms, png files
    """
    N = S.shape[0]
    routes = np.zeros((N, k), dtype=np.int32)

    # Dump folder for PBM -> PNG
    png_files = []
    if dump_prefix:
        from tempfile import TemporaryDirectory
        tmp_dir = Path(dump_prefix)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = None

    # Run routing
    t0 = time.time()
    phase_router_func(S, T, k, routes, dump=bool(tmp_dir), prefix=str(tmp_dir) if tmp_dir else "", validate=validate)
    t1 = time.time()
    routing_time_ms = (t1 - t0) * 1000

    # Convert PBMs to PNGs if available
    if tmp_dir:
        pbm_files = sorted(tmp_dir.glob("*.pbm"))
        png_files = convert_pbm_to_png(pbm_files, invert=True, png_folder=tmp_dir / "png")

    # Collect metrics
    metrics = collect_metrics(routes, S, T, k)
    metrics["routing_time_ms"] = routing_time_ms

    return metrics, routes, png_files


# -----------------------------
# Matrix generation
# -----------------------------

def generate_random_binary_matrices(N, k_max, seed_S=42, seed_T=123):
    rng_S = np.random.default_rng(seed_S)
    rng_T = np.random.default_rng(seed_T)

    row_counts_S = rng_S.integers(1, k_max + 1, size=N)
    row_counts_T = rng_T.integers(1, k_max + 1, size=N)

    S = np.zeros((N, N), dtype=np.uint8)
    T = np.zeros((N, N), dtype=np.uint8)

    for i in range(N):
        S[i, rng_S.choice(N, size=row_counts_S[i], replace=False)] = 1
        T[i, rng_T.choice(N, size=row_counts_T[i], replace=False)] = 1

    return S, T


# -----------------------------
# Scaling study
# -----------------------------

def scaling_study(N_values, k_values, phase_router_func, repeats=3):
    """
    Evaluate runtime and statistics across matrix sizes and k values
    Returns a dict of results
    """
    results = {}
    for N in N_values:
        for k in k_values:
            metrics_list = []
            for _ in range(repeats):
                S, T = generate_random_binary_matrices(N, k, seed_S=None, seed_T=None)
                metrics, _, _ = run_routing_experiment(S, T, k, phase_router_func, dump_prefix=None, validate=True)
                metrics_list.append(metrics)
            # Aggregate metrics
            agg = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
            results[(N, k)] = agg
            print(f"N={N}, k={k} -> routing_time={agg['routing_time_ms']:.2f} ms, fill_ratio={agg['fill_ratio']:.3f}")
    return results


# -----------------------------
# Visual evaluation
# -----------------------------

def visual_inspection_example(N=256, k=32, output_folder="dump/visual"):
    """
    Generate small example matrices for paper figures
    """
    S, T = generate_random_binary_matrices(N, k, seed_S=42, seed_T=123)
    metrics, routes, png_files = run_routing_experiment(S, T, k, router.pack_and_route, dump_prefix=output_folder)
    print(f"Visual inspection example completed, routing_time={metrics['routing_time_ms']:.2f} ms")
    return S, T, routes, png_files


# -----------------------------
# Reproducibility test
# -----------------------------

def reproducibility_test(N=1024, k=64, runs=3):
    """
    Verify fixed-seed reproducibility
    """
    metrics_list = []
    S_seeded, T_seeded = generate_random_binary_matrices(N, k, seed_S=42, seed_T=123)
    for i in range(runs):
        metrics, routes, _ = run_routing_experiment(S_seeded.copy(), T_seeded.copy(), k, router.pack_and_route)
        metrics_list.append((metrics, routes.copy()))

    # Check equality across runs
    all_equal = all(np.array_equal(routes, metrics_list[0][1]) for _, routes in metrics_list)
    print(f"Reproducibility check (fixed seed): {all_equal}")
    return all_equal
