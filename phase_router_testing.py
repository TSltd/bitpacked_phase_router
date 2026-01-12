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

import json
from router import run_multiphase_router  # use the new multiphase router
from router import dump_routes_image, validate_routes



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





def run_routing_experiment(S_bits_np, T_bits_np, k, dump_prefix=None, validate=False):
    """
    Run a full multiphase routing experiment on bitpacked matrices S and T.

    Args:
        S_bits_np (np.ndarray): Source bitpacked matrix (N x NB_words) dtype=uint64
        T_bits_np (np.ndarray): Target bitpacked matrix (N x NB_words) dtype=uint64
        k (int): Maximum number of routes per row
        dump_prefix (str, optional): Prefix for image dump output
        validate (bool, optional): Whether to validate routes after each phase

    Returns:
        metrics (dict): Dictionary with routing statistics (total bits, routes per phase, etc.)
        routes (np.ndarray): Final routing matrix (N x k)
        png_files (list[str]): List of dumped image file paths
    """
    N = S_bits_np.shape[0]
    NB_words = S_bits_np.shape[1]

    # Prepare routes array
    routes = np.full((N, k), -1, dtype=np.int32)

    png_files = []
    metrics = {
        "total_routes": 0,
        "phases": 0,
        "routes_per_phase": []
    }

    start_time = time.time()

    # Run multiphase router
    try:
        all_routes = run_multiphase_router(S_bits_np, T_bits_np, N, dump_prefix or "")
    except Exception as e:
        raise RuntimeError(f"Multipase router failed: {e}")

    # Fill the routes array, phase by phase
    row_idx = 0
    for phase_idx, phase_routes in enumerate(all_routes, start=1):
        phase_routes = np.array(phase_routes, dtype=np.int32)

        # Reshape if 1D to (-1, k)
        if phase_routes.ndim == 1:
            adaptive_k = k
        if len(phase_routes) < k:
            print(f"Phase {phase_idx}: fewer routes ({len(phase_routes)}) than k={k}, reducing k to {len(phase_routes)}")
            adaptive_k = len(phase_routes)
        if len(phase_routes) % adaptive_k != 0:
            # pad with -1 to make divisible
            pad_len = adaptive_k - (len(phase_routes) % adaptive_k)
            phase_routes = np.concatenate([phase_routes, -np.ones(pad_len, dtype=phase_routes.dtype)])
            print(f"Phase {phase_idx}: padded {pad_len} entries to make divisible by adaptive k={adaptive_k}")

        phase_routes = phase_routes.reshape(-1, adaptive_k)

        rows_to_fill = min(phase_routes.shape[0], N - row_idx)
        routes[row_idx:row_idx+rows_to_fill, :phase_routes.shape[1]] = phase_routes[:rows_to_fill, :]
        row_idx += rows_to_fill

        # Update metrics
        metrics["phases"] += 1
        metrics["routes_per_phase"].append(phase_routes.shape[0])
        metrics["total_routes"] += phase_routes.shape[0] * phase_routes.shape[1]

        # Dump route image for this phase if dump_prefix is provided
        if dump_prefix is not None:
            png_file = f"{dump_prefix}/phase_{phase_idx}.png"
            dump_routes_image(phase_routes, png_file, phase_routes.shape[1])
            png_files.append(png_file)

        # Optional validation after each phase
        if validate:
            valid = validate_routes(S_bits_np, T_bits_np, routes[:row_idx, :])
            if not valid:
                print(f"WARNING: Validation failed at phase {phase_idx}")

    elapsed = time.time() - start_time
    print(f"Completed routing in {metrics['phases']} phases, total routes={metrics['total_routes']}, time={elapsed:.2f}s")

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

def scaling_study(N_values, k_values, repeats=3):
    """
    Evaluate runtime and statistics across matrix sizes and k values
    using the adaptive multiphase router.
    Returns a dict of results
    """
    results = {}
    for N in N_values:
        for k in k_values:
            metrics_list = []
            for _ in range(repeats):
                S, T = generate_random_binary_matrices(N, k, seed_S=None, seed_T=None)
                metrics, _, _ = run_routing_experiment(S, T, k, dump_prefix=None, validate=True)
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
    Verify fixed-seed reproducibility using the adaptive multiphase router
    """
    metrics_list = []
    S_seeded, T_seeded = generate_random_binary_matrices(N, k, seed_S=42, seed_T=123)
    for i in range(runs):
        metrics, routes, _ = run_routing_experiment(S_seeded.copy(), T_seeded.copy(), k)
        metrics_list.append((metrics, routes.copy()))

    # Check equality across runs
    all_equal = all(np.array_equal(routes, metrics_list[0][1]) for _, routes in metrics_list)
    print(f"Reproducibility check (fixed seed): {all_equal}")
    return all_equal

def run_scaling_experiment(
    N_values, k_values,
    output_root="scaling_results",
    dump_png_for_small_N=True,
    validate=True,
    max_png_N=512,
    repeats=3
):
    """
    Run a full batch of scaling experiments across multiple N and k values
    using the adaptive multiphase router.
    Saves S/T matrices, routes, metrics JSON, and PBM->PNG for small N.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {}

    for N in N_values:
        for k in k_values:
            metrics_list = []
            for rep in range(repeats):
                S, T = generate_random_binary_matrices(N, k, seed_S=None, seed_T=None)
                run_folder = output_root / f"N{N}_k{k}_rep{rep}"
                run_folder.mkdir(parents=True, exist_ok=True)
                np.save(run_folder / "S.npy", S)
                np.save(run_folder / "T.npy", T)

                dump_prefix = run_folder if dump_png_for_small_N and N <= max_png_N else None

                metrics, routes, png_files = run_routing_experiment(
                    S, T, k, dump_prefix=dump_prefix, validate=validate
                )
                metrics_list.append(metrics)

                np.save(run_folder / "routes.npy", routes)
                with open(run_folder / "metrics.json", "w") as f:
                    json.dump(metrics_to_json_serializable(metrics), f, indent=4)

                if dump_prefix and png_files:
                    png_folder = run_folder / "png"
                    png_folder.mkdir(exist_ok=True)
                    for png in png_files:
                        dst = png_folder / png.name
                        png.rename(dst)

            # Aggregate metrics across repeats
            agg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
            summary[(N, k)] = agg_metrics

            print(f"[Scaling] N={N}, k={k} -> routing_time={agg_metrics['routing_time_ms']:.2f} ms, fill_ratio={agg_metrics['fill_ratio']:.3f}")
            with open(output_root / f"metrics_N{N}_k{k}.json", "w") as f:
                json.dump(metrics_to_json_serializable(agg_metrics), f, indent=4)

    summary_file = output_root / "summary.json"
    with open(summary_file, "w") as f:
        json.dump({f"N{N}_k{k}": metrics_to_json_serializable(v) for (N, k), v in summary.items()}, f, indent=4)

    print(f"Scaling experiment complete. All results saved in '{output_root}'")
    return summary

def metrics_to_json_serializable(x):
    """
    Recursively convert numpy types to native Python so JSON can serialize them.
    """
    import numpy as np

    if isinstance(x, dict):
        return {k: metrics_to_json_serializable(v) for k, v in x.items()}

    if isinstance(x, list):
        return [metrics_to_json_serializable(v) for v in x]

    if isinstance(x, tuple):
        return [metrics_to_json_serializable(v) for v in x]

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        return float(x)

    if isinstance(x, (np.bool_,)):
        return bool(x)

    if isinstance(x, bool):
        return bool(x)

    return x

