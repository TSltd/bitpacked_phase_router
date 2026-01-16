"""
Phase Router Test Matrix
-----------------------

Systematic stress tests for the bit-packed phase router, including:
- Row-degree extremes
- Column-target stress
- Seed reproducibility
- Monte Carlo Mean Load
- Large-scale performance
- Adversarial two-phase composability
- Edge-Case k Values
- Structured Sparse Patterns
- Phase Rotation Boundaries
- Hash Router Comparison

"""

import numpy as np
import time
import os
from pathlib import Path
import csv
# import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import resource
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
import router

try:
    import psutil
except ImportError:
    psutil = None

# ============================================================================
# CLI / Globals
# ============================================================================

parser = argparse.ArgumentParser(description="Phase Router Test Matrix")
parser.add_argument("--skip-plots", action="store_true", help="Disable plotting")
args = parser.parse_args()
SKIP_PLOTS = args.skip_plots

out = Path("test_output")
out.mkdir(exist_ok=True)
(out / "plots").mkdir(exist_ok=True)

# ============================================================================
# Utilities
# ============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg} | RSS_PEAK={mem_gb():.2f} GB", flush=True)

def mem_gb() -> float:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss_kb / (1024**3)
    return rss_kb / (1024**2)

def write_csv(rows, headers, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_markdown_table(rows, headers, path: Path):
    import platform
    with open(path, "w") as f:
        # Table
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")
        # System info
        f.write("\n**System Specs:**\n")
        f.write(f"- OS: {platform.system()} {platform.release()}\n")
        f.write(f"- Architecture: {platform.machine()}\n")
        f.write(f"- Logical CPUs: {os.cpu_count()}\n")
        if psutil:
            f.write(f"- Physical cores: {psutil.cpu_count(logical=False)}\n")
            freq = psutil.cpu_freq()
            if freq:
                f.write(f"- CPU freq: {freq.max:.1f} MHz\n")
            f.write(f"- Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")

def flatten_dict(d, parent_key="", sep="__"):
    """
    Recursively flattens nested dictionaries.
    Example:
        {"a": {"b": 1}, "c": 2}
        -> {"a__b": 1, "c": 2}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v

    return items

def write_markdown_per_test(results, path: Path):
    import platform

    with open(path, "w") as f:
        # --------------------------------------------------
        # Title & intro
        # --------------------------------------------------
        f.write("# Phase Router Test Matrix\n\n")
        f.write(
            "Systematic stress tests for the bit-packed phase router, including:\n"
            "- Row-degree extremes\n"
            "- Column-target stress\n"
            "- Seed reproducibility\n"
            "- Monte Carlo Mean Load\n"
            "- Large-scale performance\n"
            "- Adversarial two-phase composability\n"
            "- Edge-Case k Values\n"
            "- Structured Sparse Patterns\n"
            "- Phase Rotation Boundaries\n"
            "- Hash Router Comparison\n\n"
        )

        # --------------------------------------------------
        # Per-test tables
        # --------------------------------------------------
        for r in results:
            test_name = r["test"]
            result = r["result"]

            f.write(f"## {test_name}\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")

            if isinstance(result, dict):
                flat = flatten_dict(result)

                for key, value in flat.items():
                    if isinstance(value, bool):
                        val = str(value)
                    elif isinstance(value, int):
                        val = str(value)
                    elif isinstance(value, float):
                        val = f"{value:.3f}"
                    elif value is None:
                        val = ""
                    else:
                        continue  # skip non-scalars

                    f.write(f"| `{key}` | {val} |\n")
            else:
                f.write("| result | *(non-dict result)* |\n")

            f.write("\n")

        # --------------------------------------------------
        # System info (once, at end)
        # --------------------------------------------------
        f.write("---\n\n")
        f.write("### System Specs\n\n")
        f.write(f"- OS: {platform.system()} {platform.release()}\n")
        f.write(f"- Architecture: {platform.machine()}\n")
        f.write(f"- Logical CPUs: {os.cpu_count()}\n")
        if psutil:
            f.write(f"- Physical cores: {psutil.cpu_count(logical=False)}\n")
            freq = psutil.cpu_freq()
            if freq:
                f.write(f"- CPU freq: {freq.max:.1f} MHz\n")
            f.write(f"- Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")


def plot_column_loads(N, k, load_phase, load_hash, out_path, prefix="column_load"):
    if SKIP_PLOTS:
        return
    def save_hist(load, label, fname):
        bins = np.arange(0, load.max() + 2) - 0.5
        skew = load.max() / (load.mean() + 1e-9)
        plt.figure(figsize=(8,5))
        plt.hist(load, bins=bins, log=True)
        plt.xlabel("Column load")
        plt.ylabel("Number of columns (log)")
        plt.title(f"{label} | N={N}, k={k}, skew={skew:.2f}")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(out_path / fname, dpi=300)
        plt.close()
    save_hist(load_phase, "Phase Router", f"{prefix}__phase__N{N}__k{k}.png")
    save_hist(load_hash, "Hash Router", f"{prefix}__hash__N{N}__k{k}.png")

# ============================================================================
# Matrix Generation
# ============================================================================

def generate_uniform_k_matrices(N, k, seed=42):
    rng = np.random.default_rng(seed)
    M = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        M[i, rng.choice(N, k, replace=False)] = 1
    return M

def build_adversarial_S2(routes1, N, k, buckets=128, seed=123):
    rng = np.random.default_rng(seed)
    S2 = np.zeros((N, N), dtype=np.uint8)
    bucket_cols = [rng.choice(N, k, replace=False) for _ in range(buckets)]
    for i in range(N):
        h = np.sum(routes1[i][routes1[i] >= 0]) % buckets
        cols = bucket_cols[h]
        S2[i, cols] = 1
    return S2

def hash_router(S, T, k, seed=0):
    N = S.shape[0]
    rng = np.random.default_rng(seed)
    routes = -np.ones((N, k), dtype=np.int32)
    for i in range(N):
        candidates = np.where(S[i] & T[i])[0]
        if len(candidates) > 0:
            rng.shuffle(candidates)
            chosen = candidates[:k]
            routes[i, :len(chosen)] = chosen
    return routes

def column_loads_from_routes(routes, N):
    counts = np.zeros(N, dtype=int)
    for r in routes:
        for c in r:
            if c >= 0:
                counts[c] += 1
    return counts

def compute_column_statistics(routes, N):
    counts = column_loads_from_routes(routes, N)
    mean = counts.mean()
    return {
        "col_min": int(counts.min()),
        "col_max": int(counts.max()),
        "col_mean": float(mean),
        "col_std": float(counts.std()),
        "col_skew": float(counts.max() / (mean + 1e-9))
    }

# ============================================================================
# Test Scenarios
# ============================================================================

def test_row_degree_extremes(N):
    """Mix of very sparse and very dense rows"""
    log("Running Row-Degree Extremes Test")
    k_per_row = [1]*(N//2) + [min(N, N//2)]*(N//2)
    S = np.zeros((N, N), dtype=np.uint8)
    T = np.zeros((N, N), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for i in range(N):
        k_row = k_per_row[i]
        S[i, rng.choice(N, k_row, replace=False)] = 1
        T[i, rng.choice(N, k_row, replace=False)] = 1
    routes = np.empty((N, max(k_per_row)), dtype=np.int32)
    t0 = time.time()
    router.pack_and_route(S, T, max(k_per_row), routes, dump=False, validate=False)
    t_ms = (time.time() - t0) * 1000
    stats = compute_column_statistics(routes, N)
    log(f"Row-degree extremes | max load={stats['col_max']} skew={stats['col_skew']:.2f} time={t_ms:.1f} ms")
    return stats

def test_column_target_stress(N, k):
    """Highly uneven column degrees"""
    log("Running Column-Target Stress Test")
    S = generate_uniform_k_matrices(N, k, seed=1)
    T = generate_uniform_k_matrices(N, k, seed=2)
    # Inject a heavy column
    T[:, 0] = 1
    routes = np.empty((N, k), dtype=np.int32)
    t0 = time.time()
    router.pack_and_route(S, T, k, routes, dump=False, validate=False)
    t_ms = (time.time() - t0) * 1000
    stats = compute_column_statistics(routes, N)
    log(f"Column-target stress | max load={stats['col_max']} skew={stats['col_skew']:.2f} time={t_ms:.1f} ms")
    return stats

def test_seed_reproducibility(N, k):
    log("Running Seed Reproducibility Test")
    S = generate_uniform_k_matrices(N, k, seed=1)
    T = generate_uniform_k_matrices(N, k, seed=2)
    routes1 = np.empty((N, k), dtype=np.int32)
    routes2 = np.empty((N, k), dtype=np.int32)

    seed = 12345  # fixed for reproducibility

    # Run twice with the same seed
    router.pack_and_route(S, T, k, routes1, seed=seed)
    router.pack_and_route(S, T, k, routes2, seed=seed)

    identical = np.array_equal(routes1, routes2)
    log(f"Seed reproducibility test: identical={identical}")
    return {"identical": identical}


def test_monte_carlo_mean_load(N, k, seeds=10):
    log(f"Running Monte Carlo Mean Load Test | seeds={seeds}")
    col_sums = np.zeros(N, dtype=float)
    for s in range(seeds):
        S = generate_uniform_k_matrices(N, k, seed=s)
        T = generate_uniform_k_matrices(N, k, seed=s+100)
        routes = np.empty((N, k), dtype=np.int32)
        router.pack_and_route(S, T, k, routes, dump=False, validate=False)
        col_sums += column_loads_from_routes(routes, N)
    col_sums /= seeds
    mean = col_sums.mean()
    skew = col_sums.max() / (mean + 1e-9)
    log(f"Monte Carlo Mean Load | mean={mean:.2f} max/skew={skew:.2f}")
    return {"mean_load": mean, "skew": skew}

def test_large_scale(N, k):
    log(f"Running Large-Scale Test N={N}, k={k}")
    S = generate_uniform_k_matrices(N, k, seed=1)
    T = generate_uniform_k_matrices(N, k, seed=2)
    routes = np.empty((N, k), dtype=np.int32)
    t0 = time.time()
    router.pack_and_route(S, T, k, routes, dump=False, validate=False)
    t_ms = (time.time() - t0) * 1000
    stats = compute_column_statistics(routes, N)
    log(f"Large-scale test | max load={stats['col_max']} skew={stats['col_skew']:.2f} time={t_ms:.1f} ms")
    return stats

def test_adversarial_two_phase(N, k):
    """Use Phase 1 -> adversarial Phase 2 construction"""
    log(f"Running Adversarial Two-Phase Test N={N}, k={k}")
    S1 = generate_uniform_k_matrices(N, k, seed=1)
    T1 = generate_uniform_k_matrices(N, k, seed=2)
    routes1 = np.empty((N, k), dtype=np.int32)
    router.pack_and_route(S1, T1, k, routes1, dump=False, validate=False)
    S2 = build_adversarial_S2(routes1, N, k)
    T2 = generate_uniform_k_matrices(N, k, seed=3)
    routes2 = np.empty((N, k), dtype=np.int32)
    router.pack_and_route(S2, T2, k, routes2, dump=False, validate=False)
    stats = compute_column_statistics(routes2, N)
    log(f"Adversarial two-phase | max load={stats['col_max']} skew={stats['col_skew']:.2f}")
    return stats

def test_edge_case_k_values(N):
    log("Running Edge-Case k Tests (k=1, k=N)")

    results = {}

    for k in [1, N]:
        S = generate_uniform_k_matrices(N, min(k, N), seed=10)
        T = generate_uniform_k_matrices(N, min(k, N), seed=11)

        routes = np.empty((N, min(k, N)), dtype=np.int32)
        t0 = time.time()
        router.pack_and_route(S, T, min(k, N), routes, dump=False, validate=False)
        t_ms = (time.time() - t0) * 1000

        stats = compute_column_statistics(routes, N)
        results[f"k={k}"] = {
            "col_max": stats["col_max"],
            "col_skew": stats["col_skew"],
            "time_ms": t_ms
        }

        log(
            f"Edge k={k} | max load={stats['col_max']} "
            f"skew={stats['col_skew']:.2f} time={t_ms:.1f} ms"
        )

        del routes, S, T

    return results

def test_structured_sparse_patterns(N, k):
    log("Running Structured Sparse Pattern Tests")

    results = {}

    # Diagonal
    S = np.zeros((N, N), dtype=np.uint8)
    T = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        S[i, i] = 1
        T[i, i] = 1

    routes = np.empty((N, k), dtype=np.int32)
    router.pack_and_route(S, T, k, routes, dump=False, validate=False)
    stats_diag = compute_column_statistics(routes, N)
    results["diagonal"] = stats_diag

    log(
        f"Diagonal | max load={stats_diag['col_max']} "
        f"skew={stats_diag['col_skew']:.2f}"
    )

    del routes

    # Checkerboard
    S.fill(0)
    T.fill(0)
    for i in range(N):
        for j in range(0, N, 2):
            if (i + j) % 2 == 0:
                S[i, j] = 1
                T[i, j] = 1

    routes = np.empty((N, k), dtype=np.int32)
    router.pack_and_route(S, T, k, routes, dump=False, validate=False)
    stats_checker = compute_column_statistics(routes, N)
    results["checkerboard"] = stats_checker

    log(
        f"Checkerboard | max load={stats_checker['col_max']} "
        f"skew={stats_checker['col_skew']:.2f}"
    )

    del routes, S, T

    return results

def test_phase_rotation_boundaries(N, k):
    log("Running Phase Rotation Boundary Test")

    S = np.zeros((N, N), dtype=np.uint8)
    T = np.zeros((N, N), dtype=np.uint8)

    # Pack mass near the end to force wrap-around
    for i in range(N):
        S[i, max(0, N - k):N] = 1
        T[i, max(0, N - k):N] = 1

    routes = np.empty((N, k), dtype=np.int32)
    router.pack_and_route(S, T, k, routes, dump=False, validate=False)

    stats = compute_column_statistics(routes, N)

    log(
        f"Phase boundary | max load={stats['col_max']} "
        f"skew={stats['col_skew']:.2f}"
    )

    del routes, S, T
    return stats

def test_hash_router_comparison(N, k):
    log("Running Phase Router vs Hash Router Comparison")

    S = generate_uniform_k_matrices(N, k, seed=21)
    T = generate_uniform_k_matrices(N, k, seed=22)

    # Phase router
    phase_routes = np.empty((N, k), dtype=np.int32)
    router.pack_and_route(S, T, k, phase_routes, dump=False, validate=False)
    phase_stats = compute_column_statistics(phase_routes, N)

    # Hash router
    hash_routes = hash_router(S, T, k, seed=0)
    hash_stats = compute_column_statistics(hash_routes, N)

    log(
        f"Phase skew={phase_stats['col_skew']:.2f} | "
        f"Hash skew={hash_stats['col_skew']:.2f}"
    )

    if not SKIP_PLOTS:
        plot_column_loads(
            N, k,
            column_loads_from_routes(phase_routes, N),
            column_loads_from_routes(hash_routes, N),
            out / "plots",
            prefix="phase_vs_hash"
        )

    del phase_routes, hash_routes, S, T

    return {
        "phase_skew": phase_stats["col_skew"],
        "hash_skew": hash_stats["col_skew"],
        "skew_ratio": hash_stats["col_skew"] / (phase_stats["col_skew"] + 1e-9)
    }

# ----------------------------------------------------------------------
# Test registry
# Each test is (name, function)
# All functions accept (N, k) unless wrapped
# ----------------------------------------------------------------------

tests = [
    # --------------------------------------------------
    # Degree / load stress
    # --------------------------------------------------
    ("Row-Degree Extremes", lambda N, k: test_row_degree_extremes(N)),
    ("Column-Target Stress", test_column_target_stress),

    # --------------------------------------------------
    # Statistical behavior
    # --------------------------------------------------
    ("Seed Reproducibility", test_seed_reproducibility),
    ("Monte Carlo Mean Load", lambda N, k: test_monte_carlo_mean_load(N, k, seeds=10)),


    # --------------------------------------------------
    # Scaling / performance
    # --------------------------------------------------
    ("Large-Scale Performance", test_large_scale),

    # --------------------------------------------------
    # Adversarial / composability
    # --------------------------------------------------
    ("Adversarial Two-Phase", test_adversarial_two_phase),

    # --------------------------------------------------
    # Edge cases & structure
    # --------------------------------------------------
    ("Edge-Case k Values", lambda N, k: test_edge_case_k_values(N)),
    ("Structured Sparse Patterns", test_structured_sparse_patterns),
    ("Phase Rotation Boundaries", test_phase_rotation_boundaries),

    # --------------------------------------------------
    # Comparative baseline 
    # (Note: Hash router is a naïve per-row random baseline, not capacity-aware.)
    # --------------------------------------------------
    ("Hash Router Comparison", test_hash_router_comparison),
    
]

# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    N = 1024
    k = 64

    log("=== Phase Router Test Matrix ===")

    all_results = []

    for test_name, test_fn in tests:
        log(f"\n--- Running {test_name} ---")
        t0 = time.time()

        try:
            result = test_fn(N, k)
        except Exception as e:
            log(f"ERROR in {test_name}: {e}")
            result = {"error": str(e)}

        dt = (time.time() - t0) * 1000
        log(f"{test_name} completed in {dt:.1f} ms")

        all_results.append({
            "test": test_name,
            "result": result,
            "time_ms": int(dt)
        })

    # ------------------------------------------------------------------
    # Flatten results for CSV / Markdown (ONCE, after all tests)
    # ------------------------------------------------------------------

    csv_rows = []

    for r in all_results:
        row = {
            "test": r["test"],
            "time_ms": r["time_ms"],
        }

        result = r["result"]
        if isinstance(result, dict):
            flat = flatten_dict(result)

            for key, value in flat.items():
                # Allow scalar values only (CSV-safe)
                if isinstance(value, bool):
                    row[key] = value           # or int(value)
                elif isinstance(value, int):
                    row[key] = value
                elif isinstance(value, float):
                    row[key] = f"{value:.3f}"  # <-- precision control (default = 3 decimal places)
                elif value is None:
                    row[key] = ""


        csv_rows.append(row)

    # Build complete header set
    headers = sorted({key for row in csv_rows for key in row.keys()})

    write_csv(csv_rows, headers, out / "phase_router_test_matrix.csv")
    write_markdown_per_test(all_results, out / "phase_router_test_matrix.md")


    log("All tests complete")
