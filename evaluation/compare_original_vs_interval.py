"""
compare_original_vs_interval.py

Benchmark comparison between the original router.cpp and the optimized
router_interval_space.cpp.

USAGE (3 steps):

  1. Build with router.cpp (the original), then run:
       python evaluation/compare_original_vs_interval.py --tag original

  2. Change setup.py to point to router_interval_space.cpp, rebuild, then run:
       python evaluation/compare_original_vs_interval.py --tag interval

  3. Compare the two result files:
       python evaluation/compare_original_vs_interval.py --compare

Results are written to results/compare_original_vs_interval/
"""

import numpy as np
import json
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

import router

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/compare_original_vs_interval")

# Test matrix: (N, k) pairs — all N must be powers of 2
TEST_CONFIGS = [
    (256,   8),
    (256,  64),
    (512,  16),
    (512,  64),
    (1024, 16),
    (1024, 64),
    (1024, 256),
    (2048, 64),
    (2048, 256),
    (4096, 64),
    (4096, 256),
]

NUM_TRIALS  = 5        # trials per (N, k) — we report mean / std
WARMUP_RUNS = 1        # discarded warmup per config
FIXED_SEED_S = 42
FIXED_SEED_T = 123

# ---------------------------------------------------------------------------
# Matrix generation (same as phase_router_test.py)
# ---------------------------------------------------------------------------

def generate_random_binary_matrices(N: int, k_max: int,
                                    seed_S: int = 42,
                                    seed_T: int = 123):
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

# ---------------------------------------------------------------------------
# Column statistics
# ---------------------------------------------------------------------------

def compute_column_statistics(routes: np.ndarray, N: int) -> Dict[str, float]:
    col_counts = np.zeros(N, dtype=int)
    for i in range(routes.shape[0]):
        for j in routes[i]:
            if j >= 0:
                col_counts[j] += 1

    mean = float(np.mean(col_counts))
    return {
        "col_min":  int(np.min(col_counts)),
        "col_max":  int(np.max(col_counts)),
        "col_mean": mean,
        "col_std":  float(np.std(col_counts)),
        "col_skew": float(np.max(col_counts) / (mean + 1e-9)),
    }

# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def bench_one(N: int, k: int, seed: int) -> Dict:
    """Run one pack_and_route call and return timing + stats."""
    S, T = generate_random_binary_matrices(N, k, FIXED_SEED_S, FIXED_SEED_T)
    routes = np.empty((N, k), dtype=np.int32)

    # Use a deterministic router seed so results are comparable
    stats = router.pack_and_route(S, T, k, routes,
                                  dump=False, validate=False, seed=seed)

    active = int(np.sum(routes >= 0))
    col_stats = compute_column_statistics(routes, N)

    return {
        "N": N,
        "k": k,
        "seed": seed,
        "active_routes": active,
        "fill_ratio": active / (N * k),
        "routes_per_row": active / N,
        "routing_time_ms": float(stats.get("routing_time_ms", 0)),
        "packing_time_ms": float(stats.get("packing_time_ms", 0)),
        "total_time_ms":   float(stats.get("total_time_ms", 0)),
        **col_stats,
    }

# ---------------------------------------------------------------------------
# Full benchmark sweep
# ---------------------------------------------------------------------------

def run_benchmark(tag: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"bench_{tag}.json"

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK: {tag}")
    print(f"  Configs : {len(TEST_CONFIGS)}  |  Trials: {NUM_TRIALS}  |  Warmup: {WARMUP_RUNS}")
    print(f"{'=' * 70}\n")

    all_results: List[Dict] = []

    for cfg_idx, (N, k) in enumerate(TEST_CONFIGS, 1):
        print(f"[{cfg_idx}/{len(TEST_CONFIGS)}] N={N:>5}, k={k:>4}  ", end="", flush=True)

        # Warmup (discarded)
        for w in range(WARMUP_RUNS):
            bench_one(N, k, seed=9999 + w)

        # Timed trials
        trial_results = []
        for t in range(NUM_TRIALS):
            seed = 1000 + t
            r = bench_one(N, k, seed=seed)
            r["trial"] = t
            trial_results.append(r)

        # Aggregate
        routing_times = [r["routing_time_ms"] for r in trial_results]
        total_times   = [r["total_time_ms"]   for r in trial_results]
        fill_ratios   = [r["fill_ratio"]       for r in trial_results]
        col_skews     = [r["col_skew"]         for r in trial_results]

        agg = {
            "N": N,
            "k": k,
            "tag": tag,
            "num_trials": NUM_TRIALS,
            "routing_time_ms_mean": float(np.mean(routing_times)),
            "routing_time_ms_std":  float(np.std(routing_times)),
            "routing_time_ms_min":  float(np.min(routing_times)),
            "routing_time_ms_max":  float(np.max(routing_times)),
            "total_time_ms_mean":   float(np.mean(total_times)),
            "total_time_ms_std":    float(np.std(total_times)),
            "fill_ratio_mean":      float(np.mean(fill_ratios)),
            "col_skew_mean":        float(np.mean(col_skews)),
            "col_skew_std":         float(np.std(col_skews)),
            "trials": trial_results,
        }

        all_results.append(agg)

        print(f"routing {np.mean(routing_times):8.2f} ± {np.std(routing_times):6.2f} ms   "
              f"total {np.mean(total_times):8.2f} ms   "
              f"fill {np.mean(fill_ratios)*100:5.1f}%   "
              f"skew {np.mean(col_skews):5.2f}")

    # Write results
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")
    return all_results

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def run_comparison():
    orig_path     = RESULTS_DIR / "bench_original.json"
    interval_path = RESULTS_DIR / "bench_interval.json"
    report_path   = RESULTS_DIR / "comparison_report.txt"

    if not orig_path.exists():
        print(f"ERROR: {orig_path} not found.  Run with --tag original first.")
        sys.exit(1)
    if not interval_path.exists():
        print(f"ERROR: {interval_path} not found.  Run with --tag interval first.")
        sys.exit(1)

    with open(orig_path) as f:
        orig = json.load(f)
    with open(interval_path) as f:
        intv = json.load(f)

    # Index by (N, k)
    orig_map = {(r["N"], r["k"]): r for r in orig}
    intv_map = {(r["N"], r["k"]): r for r in intv}

    lines: List[str] = []

    def log(msg=""):
        lines.append(msg)
        print(msg)

    log("=" * 90)
    log("  COMPARISON:  original  vs  interval-space")
    log("=" * 90)
    log()
    log(f"{'N':>6}  {'k':>5}  "
        f"{'orig (ms)':>12}  {'intv (ms)':>12}  {'speedup':>8}  "
        f"{'orig skew':>10}  {'intv skew':>10}  "
        f"{'orig fill%':>10}  {'intv fill%':>10}")
    log("-" * 90)

    speedups = []

    all_keys = sorted(set(list(orig_map.keys()) + list(intv_map.keys())))

    for key in all_keys:
        o = orig_map.get(key)
        i = intv_map.get(key)
        if not o or not i:
            log(f"{key[0]:>6}  {key[1]:>5}  -- missing data for one version --")
            continue

        o_rt = o["routing_time_ms_mean"]
        i_rt = i["routing_time_ms_mean"]
        sp   = o_rt / i_rt if i_rt > 0 else float("inf")
        speedups.append(sp)

        log(f"{key[0]:>6}  {key[1]:>5}  "
            f"{o_rt:>10.2f}ms  {i_rt:>10.2f}ms  {sp:>7.2f}x  "
            f"{o['col_skew_mean']:>10.2f}  {i['col_skew_mean']:>10.2f}  "
            f"{o['fill_ratio_mean']*100:>9.1f}%  {i['fill_ratio_mean']*100:>9.1f}%")

    log("-" * 90)
    if speedups:
        log(f"\nGeometric mean speedup (routing): {np.exp(np.mean(np.log(speedups))):.2f}x")
        log(f"Arithmetic mean speedup:          {np.mean(speedups):.2f}x")
        log(f"Min speedup:                      {np.min(speedups):.2f}x")
        log(f"Max speedup:                      {np.max(speedups):.2f}x")
    log()

    # Write report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n✓ Comparison report saved to {report_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison: original vs interval-space router")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tag", type=str,
                       help="Run benchmark and tag results (e.g. 'original' or 'interval')")
    group.add_argument("--compare", action="store_true",
                       help="Compare bench_original.json vs bench_interval.json")
    args = parser.parse_args()

    if args.compare:
        run_comparison()
    else:
        run_benchmark(args.tag)


if __name__ == "__main__":
    main()
