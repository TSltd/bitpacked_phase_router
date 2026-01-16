
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

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import router
import matplotlib.pyplot as plt

from datetime import datetime

import os
import resource

def mem_gb() -> float:
    """Return resident memory in GB."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes
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
    """Recursively convert NumPy arrays to lists for JSON serialization."""
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
# Hash Router (Baseline)
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
# Adversarial Fan-Out Amplification (KEY)
# ============================================================================

def build_adversarial_S2(
    routes1: np.ndarray,
    N: int,
    k: int,
    buckets: int = 128,
    seed: int = 123
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S2 = np.zeros((N, N), dtype=np.uint8)

    bucket_cols = [
        rng.choice(N, k, replace=False)
        for _ in range(buckets)
    ]

    for i in range(N):
        h = np.sum(routes1[i][routes1[i] >= 0]) % buckets
        cols = bucket_cols[h]
        S2[i, cols] = 1

    return S2


# ============================================================================
# Two-Phase Adversarial Test (KILLER TEST)
# ============================================================================

def run_two_phase_adversarial_test(N: int, k: int):
    log(f"Starting two-phase adversarial test: N={N}, k={k}")

    log("Generating Phase 1 matrices")
    S1 = generate_uniform_k_matrices(N, k, seed=1)
    T1 = generate_uniform_k_matrices(N, k, seed=2)

    # ---- Phase router ----
    log("Phase router: Phase 1 routing")
    routes1 = np.empty((N, k), dtype=np.int32)
    t0 = time.time()
    router.pack_and_route(S1, T1, k, routes1, dump=False, validate=False)
    t_phase1 = (time.time() - t0) * 1000

    log("Computing Phase 1 stats")
    phase1_stats = compute_column_statistics(routes1, N)

    log("Building adversarial S2")
    S2 = build_adversarial_S2(routes1, N, k)

    # routes1 no longer needed
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

   # ---- Hash router ----
    log("Hash router: Phase 1 routing")

    S1 = generate_uniform_k_matrices(N, k, seed=1)
    T1 = generate_uniform_k_matrices(N, k, seed=2)

    t0 = time.time()
    hash1 = hash_router(S1, T1, k)
    t_hash1 = (time.time() - t0) * 1000

    hash1_stats = compute_column_statistics(hash1, N)

    log("Building adversarial S2 (hash)")
    S2_hash = build_adversarial_S2(hash1, N, k, seed=999)

    # Phase 1 hash data no longer needed
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

    log(
        f"Completed k={k} | "
        f"phase2 skew={phase2_stats['col_skew']:.2f}, "
        f"hash2 skew={hash2_stats['col_skew']:.2f}"
    )

    log("Plotting column loads...")
    # ---------- Plot ----------
    plot_column_loads(
        N, k,
        load_phase,
        load_hash,
        out / f"phase2_load_N{N}_k{k}.png"
    )

    log("Plots completed.")
    del load_phase, load_hash
    log("Released column load arrays")

    return {
        "k": k,
        "phase_router": {
            "phase1": phase1_stats,
            "phase2": phase2_stats,
            "time_ms": {
                "phase1": t_phase1,
                "phase2": t_phase2
            }
        },
        "hash_router": {
            "phase2": hash2_stats,
            "time_ms": {
                "phase1": t_hash1,
                "phase2": t_hash2
        }
    }

    }

# ============================================================================
# Plotting
# ============================================================================

def plot_column_loads(
    N: int,
    k: int,
    load_phase: np.ndarray,
    load_hash: np.ndarray,
    output_path: Path
):
    max_load = max(load_phase.max(), load_hash.max())
    bins = np.arange(0, max_load + 2) - 0.5

    max_phase = load_phase.max()
    skew_phase = max_phase / (load_phase.mean() + 1e-9)
    max_hash = load_hash.max()
    skew_hash = max_hash / (load_hash.mean() + 1e-9)

    plt.figure(figsize=(10, 6))
    plt.hist(load_phase, bins=bins, alpha=0.7, label="Phase Router", log=True)
    plt.hist(load_hash, bins=bins, alpha=0.5, label="Hash Router", log=True)

    plt.text(max_phase, 10, f"Max={max_phase}\nSkew={skew_phase:.2f}",
             ha="right", va="bottom")
    plt.text(max_hash, 10, f"Max={max_hash}\nSkew={skew_hash:.2f}",
             ha="right", va="bottom")

    plt.xlabel("Column load")
    plt.ylabel("Number of columns (log)")
    plt.title(f"Column Load Distribution: N={N}, k={k}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_phase2_column_load(N, k, routes_phase, routes_hash, output_path=None):
    def column_loads(routes):
        col_counts = np.zeros(N, dtype=int)
        for row in routes:
            for c in row:
                if c >= 0:
                    col_counts[c] += 1
        return col_counts

    load_phase = column_loads(routes_phase)
    load_hash = column_loads(routes_hash)

    # Determine bin edges dynamically
    max_load = max(load_phase.max(), load_hash.max())
    bins = np.arange(0, max_load + 2) - 0.5

    # Compute statistics for annotation
    max_phase, skew_phase = load_phase.max(), load_phase.max() / (load_phase.mean() + 1e-9)
    max_hash, skew_hash = load_hash.max(), load_hash.max() / (load_hash.mean() + 1e-9)

    plt.figure(figsize=(10, 6))
    plt.hist(load_phase, bins=bins, alpha=0.7, label='Phase Router', color='steelblue', log=True)
    plt.hist(load_hash, bins=bins, alpha=0.5, label='Hash Router', color='crimson', log=True)

    # Annotate Phase Router
    plt.text(
        max_phase, 10,  # y-position small enough for log scale
        f'Max={max_phase}\nSkew={skew_phase:.2f}',
        color='steelblue', fontsize=10, ha='right', va='bottom'
    )

    # Annotate Hash Router
    plt.text(
        max_hash, 10,
        f'Max={max_hash}\nSkew={skew_hash:.2f}',
        color='crimson', fontsize=10, ha='right', va='bottom'
    )

    plt.xlabel("Column load (number of incoming routes)")
    plt.ylabel("Number of columns (log scale)")
    plt.title(f"Phase 2 Column Load Distribution: N={N}, k={k}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Dynamic x-axis limits for clarity
    x_min = 0
    x_max = max(max_phase * 1.1, max_hash * 1.1)
    plt.xlim(x_min, x_max)

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Figure saved to {output_path}")
    plt.close()

def plot_routing_time(results, output_path=None):
    ks = [r["k"] for r in results]

    # Sum Phase 1 + Phase 2 times
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
        print(f"Timing figure saved to {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    N = 16000
    ks_single = [16, 64, 256, 1024, 4096, 12800]
    ks_two = [16, 64, 256, 1024, 4096, 12800]

    out = Path("test_output")
    out.mkdir(exist_ok=True)

    all_results = {
        "single_phase": [],
        "two_phase_adversarial": []
    }

    # ----------------------------
    # Single-Phase Sweep
    # ----------------------------
    print("\n=== Single-phase stress sweep ===")
    for k in ks_single:
        routes, stats, fill, t_ms, S, T = run_single_test(N, k, 42, 123)

        t0 = time.time()
        hash_routes = hash_router(S, T, k)
        hash_t_ms = (time.time() - t0) * 1000

        hash_stats = compute_column_statistics(hash_routes, N)
        hash_fill = compute_fill_metrics(hash_routes, N, k)

        print(
            f"k={k:5d} | phase skew={stats['col_skew']:.2f}, "
            f"hash skew={hash_stats['col_skew']:.2f}, "
            f"time={t_ms:.1f} ms"
        )

        all_results["single_phase"].append({
            "k": k,
            "phase_router": {
                "stats": stats,
                "fill": fill,
                "time_ms": t_ms
            },
            "hash_router": {
                "stats": hash_stats,
                "fill": hash_fill,
                "time_ms": hash_t_ms
            }
        })

        # Single-phase load visualization (OK to keep)
        plot_phase2_column_load(
            N, k,
            routes,
            hash_routes,
            output_path=out / f"single_phase_load_N{N}_k{k}.png"
        )

        # Free memory aggressively
        del routes, hash_routes, S, T

    # ----------------------------
    # Two-Phase Adversarial Test
    # ----------------------------
    print("\n=== Two-phase adversarial composability test ===")
    for k in ks_two:
        result = run_two_phase_adversarial_test(N, k)
        all_results["two_phase_adversarial"].append(result)

        # NO plotting here
        # NO route access here
        # run_two_phase_adversarial_test already saved plots

    # ----------------------------
    # Save Results
    # ----------------------------
    json_path = out / "stress_test_results.json"
    with open(json_path, "w") as f:
        json.dump(make_json_serializable(all_results), f, indent=2)
    print(f"All metrics saved to {json_path}")

    # Routing time vs k (Phase 1 + Phase 2)
    plot_routing_time(
        all_results["two_phase_adversarial"],
        output_path=out / "phase1_plus_phase2_routing_time.png"
    )

    print("\n✓ All tests complete")
