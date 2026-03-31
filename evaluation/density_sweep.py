# evaluation/density_sweep.py

import numpy as np
import json
import argparse
from pathlib import Path
import router

RESULTS_DIR = Path("results/density_sweep")

Ns = [4096, 8192, 16384, 32768]
densities = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

NUM_TRIALS = 10
SEED = 42


# ------------------------------------------------------------
# Controlled-density generator
# ------------------------------------------------------------

def generate_fixed_density(N, density, seed):
    rng = np.random.default_rng(seed)

    M = np.zeros((N, N), dtype=np.uint8)
    nnz_per_row = max(1, int(density * N))

    for i in range(N):
        cols = rng.choice(N, size=nnz_per_row, replace=False)
        M[i, cols] = 1

    return M


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

def bench(N, density, k):
    times = []
    fills = []

    for t in range(NUM_TRIALS):
        S = generate_fixed_density(N, density, SEED + t)
        T = generate_fixed_density(N, density, SEED + 100 + t)

        routes = np.empty((N, k), dtype=np.int32)

        stats = router.pack_and_route(S, T, k, routes, seed=1234 + t)


        times.append(stats["routing_time_ms"])
        fills.append(stats["fill_ratio"])
        

    return {
        "N": N,
        "density": density,
        "k": k,
        "routing_time_ms_mean": float(np.mean(times)),
        "routing_time_ms_std": float(np.std(times)),
        "fill_ratio_mean": float(np.mean(fills)),
        "fill_ratio": stats["fill_ratio"],
    }

def bench_equal_work(N, target_density_product, k):
    # density(S) = density(T) = sqrt(target)
    d = np.sqrt(target_density_product)

    S = generate_fixed_density(N, d, 123)
    T = generate_fixed_density(N, d, 456)

    routes = np.empty((N, k), dtype=np.int32)

    stats = router.pack_and_route(S, T, k, routes, seed=999)

    return {
        "N": N,
        "density": d,
        "target_density_product": target_density_product,
        "routing_time_ms": stats["routing_time_ms"],
        "fill_ratio": stats["fill_ratio"],
    }

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for N in Ns:
        k = 64  # keep fixed for now

        print(f"\n=== N = {N} ===")

        for d in densities:
            print(f"  density={d:.4f} ...", end="", flush=True)

            r = bench(N, d, k)
            results.append(r)

            print(f" {r['routing_time_ms_mean']:.2f} ms, fill={r['fill_ratio_mean']*100:.2f}%")

    out_path = RESULTS_DIR / "density_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ saved to {out_path}")

    # ------------------------------------------------------------
    # Equal-work experiment (needle vs haystack)
    # ------------------------------------------------------------

    print("\n" + "="*60)
    print(" EQUAL-WORK EXPERIMENT (fixed output density)")
    print("="*60)

    equal_results = []

    targets = [
        1e-5,
        5e-5,
        1e-4,
    ]

    Ns_equal = [4096, 8192, 16384, 32768]

    for target in targets:
        print(f"\n--- target output density ≈ {target} ---")

        for N in Ns_equal:
            print(f"  N={N} ...", end="", flush=True)

            r = bench_equal_work(N, target, k=64)
            equal_results.append(r)

            print(f" {r['routing_time_ms']:.2f} ms, fill={r['fill_ratio']*100:.4f}%")

    out_eq = RESULTS_DIR / "equal_work.json"
    with open(out_eq, "w") as f:
        json.dump(equal_results, f, indent=2)

    print(f"\n✓ equal-work results saved to {out_eq}")


if __name__ == "__main__":
    main()