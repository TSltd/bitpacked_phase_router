"""
Single- and multi-phase stress testing for the bit-packed phase router.
Compares against a simple hash-based router and includes a *true*
adversarial two-phase composability test where hash routing collapses.
"""

import numpy as np
import time
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import sys
import argparse
import csv
import os
import resource
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
import router

RESULTS_ROOT = Path(os.environ.get("PHASE_ROUTER_RESULTS", "results/phase_router_vs_hash")).resolve()

OUT_SINGLE = RESULTS_ROOT / "single_phase"
OUT_TWO = RESULTS_ROOT / "two_phase_adversarial"
OUT_PLOTS = RESULTS_ROOT / "plots"

OUT_SINGLE.mkdir(parents=True, exist_ok=True)
OUT_TWO.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Memory and Logging Utilities
# ============================================================================

def print_system_specs():
    import platform
    try:
        import psutil
    except ImportError:
        print("Install psutil for full system info: pip install psutil")
        psutil = None
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Logical CPUs: {os.cpu_count()}")
    if psutil:
        print(f"Physical cores: {psutil.cpu_count(logical=False)}")
        print(f"CPU max freq: {psutil.cpu_freq().max:.1f} MHz")
        print(f"Total RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")


def mem_gb() -> float:
    """Return resident memory in GB."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss_kb / (1024**3)
    return rss_kb / (1024**2)

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg} | RSS={mem_gb():.2f} GB", flush=True)

# ============================================================================
# Matrix Generation
# ============================================================================

def generate_random_binary_matrices(
    N: int,
    k_max: int,
    seed_S: Optional[int] = 42,
    seed_T: Optional[int] = 123
) -> Tuple[np.ndarray, np.ndarray]:
    rng_S = np.random.default_rng(seed_S)
    rng_T = np.random.default_rng(seed_T)
    row_counts_S = rng_S.integers(1, k_max + 1, size=N)
    row_counts_T = rng_T.integers(1, k_max + 1, size=N)
    S = np.zeros((N, N), dtype=np.uint8)
    T = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        S[i, rng_S.choice(N, row_counts_S[i], replace=False)] = 1
        T[i, rng_T.choice(N, row_counts_T[i], replace=False)] = 1
    return S, T

def generate_uniform_k_matrices(N: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        M[i, rng.choice(N, k, replace=False)] = 1
    return M

# ============================================================================
# Statistics
# ============================================================================

def compute_column_statistics(routes: np.ndarray, N: int) -> Dict[str, float]:
    col_counts = np.zeros(N, dtype=int)
    for i in range(routes.shape[0]):
        for j in routes[i]:
            if j >= 0:
                col_counts[j] += 1
    mean = np.mean(col_counts)
    return {
        "col_min": int(col_counts.min()),
        "col_max": int(col_counts.max()),
        "col_mean": float(mean),
        "col_std": float(np.std(col_counts)),
        "col_skew": float(col_counts.max() / (mean + 1e-9))
    }

def compute_fill_metrics(routes: np.ndarray, N: int, k: int) -> Dict[str, float]:
    active = np.sum(routes >= 0)
    return {
        "active_routes": int(active),
        "routes_per_row": float(active / N),
        "fill_ratio": float(active / (N * k))
    }

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj

def column_loads_from_routes(routes: np.ndarray, N: int) -> np.ndarray:
    col_counts = np.zeros(N, dtype=np.int32)
    for row in routes:
        for c in row:
            if c >= 0:
                col_counts[c] += 1
    return col_counts

# ============================================================================
# Hash Router
# ============================================================================

def hash_router(S: np.ndarray, T: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
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

# ============================================================================
# Single-Phase Test
# ============================================================================

def run_single_test(N: int, k: int, seed_S: int, seed_T: int):
    S, T = generate_random_binary_matrices(N, k, seed_S, seed_T)
    routes = np.zeros((N, k), dtype=np.int32)
    t0 = time.time()
    router.pack_and_route(S, T, k, routes, dump=False, validate=False)
    t_ms = (time.time() - t0) * 1000
    stats = compute_column_statistics(routes, N)
    fill = compute_fill_metrics(routes, N, k)
    return routes, stats, fill, t_ms, S, T

# ============================================================================
# Adversarial Phase 2 Matrix
# ============================================================================

def build_adversarial_S2(routes1: np.ndarray, N: int, k: int, buckets: int = 128, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S2 = np.zeros((N, N), dtype=np.uint8)
    bucket_cols = [rng.choice(N, k, replace=False) for _ in range(buckets)]
    for i in range(N):
        h = np.sum(routes1[i][routes1[i] >= 0]) % buckets
        cols = bucket_cols[h]
        S2[i, cols] = 1
    return S2

# ============================================================================
# Two-Phase Adversarial Test
# ============================================================================

def run_two_phase_adversarial_test(N: int, k: int):
    log(f"Starting two-phase adversarial test: N={N}, k={k}")

    # Phase 1
    log("Generating Phase 1 matrices")
    S1 = generate_uniform_k_matrices(N, k, seed=1)
    T1 = generate_uniform_k_matrices(N, k, seed=2)

    log("Phase router: Phase 1 routing")
    routes1 = np.empty((N, k), dtype=np.int32)
    t0 = time.time()
    router.pack_and_route(S1, T1, k, routes1, dump=False, validate=False)
    t_phase1 = (time.time() - t0) * 1000
    phase1_stats = compute_column_statistics(routes1, N)

    log("Building adversarial S2")
    S2 = build_adversarial_S2(routes1, N, k)
    del routes1, S1, T1
    log("Released Phase 1 data")

    log("Generating Phase 2 T2")
    T2 = generate_uniform_k_matrices(N, k, seed=3)
    log("Phase router: Phase 2 routing")
    routes2 = np.empty((N, k), dtype=np.int32)
    t0 = time.time()
    router.pack_and_route(S2, T2, k, routes2, dump=False, validate=False)
    t_phase2 = (time.time() - t0) * 1000

    load_phase = column_loads_from_routes(routes2, N)
    phase2_stats = {
        "col_min": int(load_phase.min()),
        "col_max": int(load_phase.max()),
        "col_mean": float(load_phase.mean()),
        "col_std": float(load_phase.std()),
        "col_skew": float(load_phase.max() / (load_phase.mean() + 1e-9))
    }
    del routes2, S2, T2
    log("Released Phase router Phase 2 data")

    # Hash router Phase 1
    log("Hash router: Phase 1 routing")
    S1 = generate_uniform_k_matrices(N, k, seed=1)
    T1 = generate_uniform_k_matrices(N, k, seed=2)
    t0 = time.time()
    hash1 = hash_router(S1, T1, k)
    t_hash1 = (time.time() - t0) * 1000
    hash1_stats = compute_column_statistics(hash1, N)

    log("Building adversarial S2 (hash)")
    S2_hash = build_adversarial_S2(hash1, N, k, seed=999)
    del hash1, S1, T1
    log("Released hash Phase 1 data")

    log("Hash router: Phase 2 routing")
    T2 = generate_uniform_k_matrices(N, k, seed=3)
    t0 = time.time()
    hash2 = hash_router(S2_hash, T2, k)
    t_hash2 = (time.time() - t0) * 1000
    load_hash = column_loads_from_routes(hash2, N)
    hash2_stats = {
        "col_min": int(load_hash.min()),
        "col_max": int(load_hash.max()),
        "col_mean": float(load_hash.mean()),
        "col_std": float(load_hash.std()),
        "col_skew": float(load_hash.max() / (load_hash.mean() + 1e-9))
    }
    del hash2, S2_hash, T2
    log("Released hash Phase 2 data")

    log(f"Completed k={k} | phase2 skew={phase2_stats['col_skew']:.2f}, hash2 skew={hash2_stats['col_skew']:.2f}")

    if not SKIP_PLOTS:
        log("Plotting column loads...")
        plot_column_loads(N, k, load_phase, load_hash, out_dir=OUT_PLOTS, prefix=f"phase1_column_load_k{k}")
        del load_phase, load_hash
        log("Plots completed and column load arrays released")

    return {
        "k": k,
        "phase_router": {
            "phase1": phase1_stats,
            "phase2": phase2_stats,
            "time_ms": {"phase1": t_phase1, "phase2": t_phase2}
        },
        "hash_router": {
            "phase2": hash2_stats,
            "time_ms": {"phase1": t_hash1, "phase2": t_hash2}
        }
    }

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_column_loads(N: int, k: int, load_phase: np.ndarray, load_hash: np.ndarray, out_dir: Path, prefix: str):
    out_dir.mkdir(exist_ok=True, parents=True)

    def save_hist(load, label, fname):
        max_load = load.max()
        bins = np.arange(0, max_load + 2) - 0.5
        skew = max_load / (load.mean() + 1e-9)
        plt.figure(figsize=(8, 5))
        plt.hist(load, bins=bins, log=True)
        plt.xlabel("Column load")
        plt.ylabel("Number of columns (log)")
        plt.title(f"{label} Column Load\nN={N}, k={k}, skew={skew:.2f}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=300)
        plt.close()

    save_hist(load_phase, "Phase Router", f"{prefix}__phase_router__N{N}__k{k}.png")
    save_hist(load_hash, "Hash Router", f"{prefix}__hash_router__N{N}__k{k}.png")

def plot_routing_time(results, output_path=None):
    ks = [r["k"] for r in results]
    phase_times = [r["phase_router"]["time_ms"]["phase1"] + r["phase_router"]["time_ms"]["phase2"] for r in results]
    hash_times  = [r["hash_router"]["time_ms"]["phase1"]  + r["hash_router"]["time_ms"]["phase2"]  for r in results]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, phase_times, 'o-', label='Phase Router', color='steelblue')
    plt.plot(ks, hash_times, 's--', label='Hash Router', color='crimson')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("k (routes per row)")
    plt.ylabel("Total routing time (Phase1 + Phase2, ms)")
    plt.title("End-to-End Routing Time vs k")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        log(f"Timing figure saved to {output_path}")
    plt.close()

# ============================================================================
# Markdown / CSV Output
# ============================================================================

def write_markdown_table(rows, headers, path: Path):
    import platform
    try:
        import psutil
    except ImportError:
        psutil = None

    with open(path, "w") as f:
        # Write table
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")

        # Write system info as Markdown text after table
        f.write("\n\n**System Specs:**\n\n")

        # OS / CPU / RAM
        cpu_model = platform.processor() or "Unknown"
        f.write(f"- OS: {platform.system()} {platform.release()}\n")
        f.write(f"- Architecture: {platform.machine()}\n")
        f.write(f"- CPU model: {cpu_model}\n")
        logical_threads = os.cpu_count()
        f.write(f"- Logical CPUs: {logical_threads}\n")

        if psutil:
            physical_cores = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq()
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            f.write(f"- Physical cores: {physical_cores}\n")
            if cpu_freq:
                f.write(f"- CPU max frequency: {cpu_freq.max:.1f} MHz\n")
            f.write(f"- Total RAM: {total_ram_gb:.2f} GB\n")
        else:
            f.write("- psutil not installed, detailed CPU/RAM info unavailable\n")

def write_csv(rows, headers, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def make_summary_md(
    results: dict,
    md_path: Path,
    plot_dir: Path,
    routing_time_plot: Path = None
):
    import os
    import re

    lines = []

    # -------------------------------
    # Overall Routing Time Plot
    # -------------------------------
    if routing_time_plot and routing_time_plot.exists():
        rel_path = os.path.relpath(routing_time_plot, start=md_path.parent)
        lines.append("## Overall Routing Time\n")
        lines.append(f'<img src="{rel_path}" width="600">\n')

    # -------------------------------
    # Single-Phase Table
    # -------------------------------
    lines.append("## Single-Phase Stress Sweep\n")
    lines.append(
        "| k | Phase Router Time (ms) | Active Routes | Routes/Row | "
        "Column Skew | Fill Ratio | Hash Router Time (ms) | Hash Column Skew |"
    )
    lines.append(
        "|---|-----------------------|---------------|------------|"
        "-------------|------------|----------------------|-----------------|"
    )

    for entry in results["single_phase"]:
        k = entry["k"]
        phase_time = entry["phase_router"]["time_ms"]
        active_routes = entry["phase_router"]["fill"]["active_routes"]
        routes_per_row = entry["phase_router"]["fill"]["routes_per_row"]
        col_skew = entry["phase_router"]["stats"]["col_skew"]
        fill_ratio = entry["phase_router"]["fill"]["fill_ratio"]

        hash_time = entry["hash_router"]["time_ms"]
        hash_skew = entry["hash_router"]["stats"]["col_skew"]

        lines.append(
            f"| {k} | {phase_time:.1f} | {active_routes} | "
            f"{routes_per_row:.4f} | {col_skew:.2f} | {fill_ratio:.5f} | "
            f"{hash_time:.1f} | {hash_skew:.2f} |"
        )

    # -------------------------------
    # Single-Phase Plots
    # -------------------------------
    lines.append("\n### Single-Phase Plots\n")
    if plot_dir.exists():
        for k_val in sorted({r["k"] for r in results["single_phase"]}):
            imgs = sorted(plot_dir.glob(f"*phase1_column_load_k{k_val}*.png"))
            if imgs:
                lines.append(f"#### k={k_val}")
                img_tags = []
                for img in imgs:
                    rel_img = os.path.relpath(img, start=md_path.parent)
                    img_tags.append(f'<img src="{rel_img}" width="250">')
                lines.append(" ".join(img_tags))
                lines.append("")

    # -------------------------------
    # Two-Phase Table
    # -------------------------------
    lines.append("\n## Two-Phase Adversarial Test\n")
    lines.append(
        "| k | Phase Router Phase 1 Time (ms) | Phase Router Phase 2 Time (ms) | "
        "Phase 2 Max Column Load | Phase 2 Column Skew | "
        "Hash Router Phase 1 Time (ms) | Hash Router Phase 2 Time (ms) | "
        "Hash Phase 2 Max Column Load | Hash Phase 2 Column Skew |"
    )
    lines.append(
        "|---|-------------------------------|-------------------------------|"
        "------------------------|------------------|"
        "-------------------------------|-------------------------------|"
        "---------------------------|-------------------------|"
    )

    for entry in results["two_phase_adversarial"]:
        k = entry["k"]

        phase1_time = entry["phase_router"]["time_ms"]["phase1"]
        phase2_time = entry["phase_router"]["time_ms"]["phase2"]
        phase2_stats = entry["phase_router"]["phase2"]
        phase2_max = phase2_stats["col_max"]
        phase2_skew = phase2_stats["col_skew"]

        hash1_time = entry["hash_router"]["time_ms"]["phase1"]
        hash2_time = entry["hash_router"]["time_ms"]["phase2"]
        hash2_stats = entry["hash_router"]["phase2"]
        hash2_max = hash2_stats["col_max"]
        hash2_skew = hash2_stats["col_skew"]

        lines.append(
            f"| {k} | {phase1_time:.1f} | {phase2_time:.1f} | "
            f"{phase2_max} | {phase2_skew:.2f} | "
            f"{hash1_time:.1f} | {hash2_time:.1f} | "
            f"{hash2_max} | {hash2_skew:.2f} |"
        )

    # -------------------------------
    # Two-Phase Plots
    # -------------------------------
    lines.append("\n### Two-Phase Plots\n")
    if plot_dir.exists():
        for k_val in sorted({r["k"] for r in results["two_phase_adversarial"]}):
            imgs = sorted(plot_dir.glob(f"*phase_router*_k{k_val}*.png"))
            if imgs:
                lines.append(f"#### k={k_val}")
                img_tags = []
                for img in imgs:
                    rel_img = os.path.relpath(img, start=md_path.parent)
                    img_tags.append(f'<img src="{rel_img}" width="250">')
                lines.append(" ".join(img_tags))
                lines.append("")

    # -------------------------------
    # Write file
    # -------------------------------
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown report written to {md_path}")

def collect_plot_paths(plot_dir: Path, all_results: dict) -> dict:
    """
    Collect PNG paths for each k in single- and two-phase tests
    and return a dictionary suitable for including in JSON.
    """
    plot_info = {"single_phase": {}, "two_phase_adversarial": {}}

    # Single-phase plots
    for entry in all_results["single_phase"]:
        k = entry["k"]
        imgs = sorted(plot_dir.glob(f"*phase1_column_load_k{k}*.png"))
        plot_info["single_phase"][k] = [str(p.relative_to(plot_dir.parent)) for p in imgs]

    # Two-phase plots (both phase router and hash)
    for entry in all_results["two_phase_adversarial"]:
        k = entry["k"]
        imgs = sorted(plot_dir.glob(f"*phase_router*_k{k}*.png"))
        plot_info["two_phase_adversarial"][k] = [str(p.relative_to(plot_dir.parent)) for p in imgs]

    # Overall routing time plot
    overall_plot = plot_dir / "phase1_plus_phase2_routing_time.png"
    if overall_plot.exists():
        plot_info["overall_routing_time"] = str(overall_plot.relative_to(plot_dir.parent))

    return plot_info


# ============================================================================
# CLI Argument
# ============================================================================

parser = argparse.ArgumentParser(description="Phase router stress tests")
parser.add_argument("--skip-plots", action="store_true", help="Disable all plotting (faster, lower memory use)")
args = parser.parse_args()
SKIP_PLOTS = args.skip_plots


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    N = 16000
    ks_single = [16, 64, 256, 1024, 4096, 12800]
    ks_two = [16, 64, 256, 1024, 4096, 12800]

    all_results = {"single_phase": [], "two_phase_adversarial": []}

    print_system_specs()

    # ===========================
    # 1. Single-Phase Sweep
    # ===========================
    log("\n=== Single-phase stress sweep ===")
    single_csv_rows = []

    for k in ks_single:
        log(f"Processing single-phase k={k} ...")
        t_start_k = time.time()

        # Phase router
        routes, stats, fill, t_ms, S, T = run_single_test(N, k, 42, 123)

        # Hash router
        t0 = time.time()
        hash_routes = hash_router(S, T, k)
        hash_t_ms = (time.time() - t0) * 1000
        hash_stats = compute_column_statistics(hash_routes, N)
        hash_fill = compute_fill_metrics(hash_routes, N, k)

        log(f"k={k:5d} | phase skew={stats['col_skew']:.2f}, hash skew={hash_stats['col_skew']:.2f}, time={t_ms:.1f} ms")

        all_results["single_phase"].append({
            "k": k,
            "phase_router": {"stats": stats, "fill": fill, "time_ms": t_ms},
            "hash_router": {"stats": hash_stats, "fill": hash_fill, "time_ms": hash_t_ms}
        })

        single_csv_rows.append({
            "k": k,
            "phase_col_max": stats["col_max"],
            "phase_col_skew": stats["col_skew"],
            "phase_fill_ratio": fill["fill_ratio"],
            "phase_time_ms": t_ms,
            "hash_col_max": hash_stats["col_max"],
            "hash_col_skew": hash_stats["col_skew"],
            "hash_fill_ratio": hash_fill["fill_ratio"],
            "hash_time_ms": hash_t_ms,
        })

        # Plot column loads if enabled
        if not SKIP_PLOTS:
            load_phase = column_loads_from_routes(routes, N)
            load_hash  = column_loads_from_routes(hash_routes, N)
            plot_column_loads(N, k, load_phase, load_hash, out_dir=OUT_PLOTS, prefix=f"phase1_column_load_k{k}")
            del load_phase, load_hash

        del routes, hash_routes, S, T
        log(f"Finished single-phase k={k} | elapsed {time.time() - t_start_k:.1f}s | RSS={mem_gb():.2f} GB")

    # Save single-phase CSV
    write_csv(single_csv_rows, [
        "k", "phase_col_max", "phase_col_skew", "phase_fill_ratio", "phase_time_ms",
        "hash_col_max", "hash_col_skew", "hash_fill_ratio", "hash_time_ms"
    ], OUT_SINGLE / "single_phase_results.csv")
    log(f"Single-phase CSV saved to {OUT_SINGLE / 'single_phase_results.csv'}")

    # ===========================
    # Save single-phase JSON with headers
    # ===========================
    json_single = {
        "title": "Single-Phase Stress Test Results",
        "description": "Single-phase stress tests comparing the phase router vs hash router",
        "date": datetime.now().isoformat(),
        "N": N,
        "ks_single": ks_single,
        "results": all_results["single_phase"],
        "plots": collect_plot_paths(OUT_PLOTS, {"single_phase": all_results["single_phase"], "two_phase_adversarial": []})["single_phase"]
    }

    with open(OUT_SINGLE / "single_phase_results.json", "w") as f:
        json.dump(make_json_serializable(json_single), f, indent=2)
    log(f"Single-phase JSON + plot metadata saved to {OUT_SINGLE / 'single_phase_results.json'}")


    # ===========================
    # 2. Two-Phase Adversarial Test
    # ===========================
    log("\n=== Two-phase adversarial composability test ===")
    two_phase_csv_rows = []

    for k in ks_two:
        log(f"Processing two-phase adversarial k={k} ...")
        t_start_k = time.time()
        result = run_two_phase_adversarial_test(N, k)
        all_results["two_phase_adversarial"].append(result)

        two_phase_csv_rows.append({
            "k": result["k"],
            "phase2_col_max": result["phase_router"]["phase2"]["col_max"],
            "phase2_col_skew": result["phase_router"]["phase2"]["col_skew"],
            "hash2_col_max": result["hash_router"]["phase2"]["col_max"],
            "hash2_col_skew": result["hash_router"]["phase2"]["col_skew"],
            "phase_time_ms": int(result["phase_router"]["time_ms"]["phase1"] + result["phase_router"]["time_ms"]["phase2"]),
            "hash_time_ms": int(result["hash_router"]["time_ms"]["phase1"] + result["hash_router"]["time_ms"]["phase2"]),
        })

        log(f"Finished two-phase k={k} | elapsed {time.time() - t_start_k:.1f}s | RSS={mem_gb():.2f} GB")

    # Save two-phase CSV
    write_csv(two_phase_csv_rows, [
        "k", "phase2_col_max", "phase2_col_skew", "hash2_col_max", "hash2_col_skew",
        "phase_time_ms", "hash_time_ms"
    ], OUT_TWO / "two_phase_adversarial_results.csv")
    log(f"Two-phase CSV saved to {OUT_TWO / 'two_phase_adversarial_results.csv'}")

    # Save Markdown table
    md_rows = []
    for r in all_results["two_phase_adversarial"]:
        md_rows.append({
            "k": r["k"],
            "phase2_max_load": r["phase_router"]["phase2"]["col_max"],
            "phase2_col_skew": f"{r['phase_router']['phase2']['col_skew']:.2f}",
            "hash2_max_load": r["hash_router"]["phase2"]["col_max"],
            "hash2_col_skew": f"{r['hash_router']['phase2']['col_skew']:.2f}",
            "phase_time_ms": int(r["phase_router"]["time_ms"]["phase1"] + r["phase_router"]["time_ms"]["phase2"]),
            "hash_time_ms": int(r["hash_router"]["time_ms"]["phase1"] + r["hash_router"]["time_ms"]["phase2"]),
        })
    write_markdown_table(md_rows, [
        "k", "phase2_max_load", "phase2_col_skew", "hash2_max_load", "hash2_col_skew", "phase_time_ms", "hash_time_ms"
    ], OUT_TWO / "two_phase_adversarial_results.md")
    log(f"Two-phase Markdown saved to {OUT_TWO / 'two_phase_adversarial_results.md'}")

    # ===========================
    # 3. Save JSON with headers/metadata
    # ===========================
    json_output = {
        "title": "Phase Router Stress Test Results",
        "description": "Single- and two-phase stress tests comparing the phase router vs hash router",
        "date": datetime.now().isoformat(),
        "N": N,
        "ks_single": ks_single,
        "ks_two": ks_two,
        "results": all_results
    }

    # ===========================
    # 4. Generate overall routing time plot
    # ===========================
    if not SKIP_PLOTS:
        routing_time_plot = OUT_PLOTS / "phase1_plus_phase2_routing_time.png"
        plot_routing_time(all_results["two_phase_adversarial"], output_path=routing_time_plot)
    else:
        routing_time_plot = None

    # Collect all plot paths for JSON metadata
    plots_metadata = collect_plot_paths(OUT_PLOTS, all_results)
    json_output["plots"] = plots_metadata

    # ===========================
    # 5. Final JSON save (single write)
    # ===========================
    with open(OUT_TWO / "two_phase_adversarial_results.json", "w") as f:
        json.dump(make_json_serializable(json_output), f, indent=2)
    log(f"JSON results + plot metadata saved to {OUT_TWO / 'two_phase_adversarial_results.json'}")

    # ===========================
    # 6. Generate Markdown summary
    # ===========================
    make_summary_md(
        all_results,
        OUT_TWO / "phase_router_summary.md",
        OUT_PLOTS,
        routing_time_plot=routing_time_plot
    )
    log("All tests complete")
                