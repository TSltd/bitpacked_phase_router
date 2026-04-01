# evaluation/density_sweep.py

import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import router

import os
print("RUNNING FILE:", os.path.abspath(__file__))

RESULTS_DIR = Path("results/density_sweep")

Ns = [4096, 8192, 16384, 32768]
densities = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
k_values = [8, 64, 512, 2048, 8192]
NUM_TRIALS = 5

SEED = 42

# Ns = [4096]
# densities = [0.01, 0.05]
# k_values = [8, 2048]
# NUM_TRIALS = 1

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
    times_route = []
    times_extract = []
    fills = []

    events_list = []
    words_list = []
    kernels = [] 

    for t in range(NUM_TRIALS):
        S = generate_fixed_density(N, density, SEED + t)
        T = generate_fixed_density(N, density, SEED + 100 + t)

        routes = np.empty((N, k), dtype=np.int32)

        stats = router.pack_and_route(S, T, k, routes, seed=1234 + t)

        times_route.append(stats["route_time_ms"])
        times_extract.append(stats["extract_time_ms"])
        fills.append(stats["fill_ratio"])

        events_list.append(stats["events"])
        words_list.append(stats["words_touched"])
        kernels.append(stats["kernel"])

    return {
        "N": N,
        "density": density,
        "k": k,
        "route_time_ms": float(np.mean(times_route)),
        "extract_time_ms": float(np.mean(times_extract)),
        "fill_ratio": float(np.mean(fills)),
        "events": float(np.mean(events_list)),
        "words_touched": float(np.mean(words_list)),
        "kernel": max(set(kernels), key=kernels.count),
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

def plot_routing_vs_N(data):
    grouped = group_by_N(data)

    Ns = []
    route_times = []

    for N, results in grouped.items():
        Ns.append(N)
        route_times.append(np.mean([r["route_time_ms"] for r in results]))

    plt.figure()
    plt.plot(Ns, route_times, marker="o")

    plt.xlabel("N")
    plt.ylabel("Routing time (ms)")
    plt.title("Routing cost vs N")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)

    plt.savefig(RESULTS_DIR / "routing_vs_N.png")

def plot_extract_vs_output(data):
    grouped = group_by_N(data)

    plt.figure()

    for N, results in grouped.items():
        fills = [r["fill_ratio"] for r in results]
        times = [r["extract_time_ms"] for r in results]

        plt.plot(fills, times, marker="o", label=f"N={N}")

    plt.xlabel("Output density")
    plt.ylabel("Extraction time (ms)")
    plt.title("Extraction vs Output Size")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)

    plt.savefig(RESULTS_DIR / "extract_vs_output.png")

def plot_normalized(data):
    grouped = group_by_N(data)

    plt.figure()

    for N, results in grouped.items():
        fills = [r["fill_ratio"] for r in results]
        times = [(r["route_time_ms"] + r["extract_time_ms"]) / N for r in results]

        plt.plot(fills, times, marker="o", label=f"N={N}")

    plt.xlabel("Output density")
    plt.ylabel("Runtime / N")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    plt.savefig(RESULTS_DIR / "normalized.png")

def plot_runtime_vs_events(data):
    grouped = group_by_N(data)

    plt.figure()

    for N, results in grouped.items():
        events = [r["events"] for r in results]
        times = [r["route_time_ms"] + r["extract_time_ms"] for r in results]

        plt.plot(events, times, marker="o", label=f"N={N}")

    plt.xlabel("Events (N * fill_ratio)")
    plt.ylabel("Total runtime (ms)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    plt.savefig(RESULTS_DIR / "runtime_vs_events.png")
def group_by_N(data):
    grouped = defaultdict(list)
    for r in data:
        grouped[r["N"]].append(r)
    return grouped

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=== DEBUG ===")
    print("FILE:", os.path.abspath(__file__))
    print("CWD:", os.getcwd())
    print("Ns =", Ns)
    print("densities =", densities)
    print("k_values =", k_values if "k_values" in globals() else "N/A")
    print("NUM_TRIALS =", NUM_TRIALS)
    print("================\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for N in Ns:

        print(f"\n=== N = {N} ===")

        for k in k_values:

            print(f"\n=== k = {k} ===")

            for d in densities:
                print(f"  density={d:.4f} ...", end="", flush=True)

                r = bench(N, d, k)
                results.append(r)

                print(f" {r['route_time_ms']:.2f} + {r['extract_time_ms']:.2f} ms, fill={r['fill_ratio']*100:.2f}%")

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

# ---------

    print("\n" + "="*60)
    print(" Routing vs N Experiment")
    print("="*60)

    plot_routing_vs_N(results)

    out_eq = RESULTS_DIR / "routing_vs_N.json"
    with open(out_eq, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Routing vs N results saved to {out_eq}")

# ----------

    print("\n" + "="*60)
    print(" Extraction vs output size Experiment")
    print("="*60)

    plot_extract_vs_output(results)

    out_eq = RESULTS_DIR / "extract_vs_output.json"
    with open(out_eq, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Extraction vs output size results saved to {out_eq}")

# -----------

    print("\n" + "="*60)
    print(" runtime / N (normalized) Experiment")
    print("="*60)

    plot_normalized(results)

    out_eq = RESULTS_DIR / "normalized.json"
    with open(out_eq, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ normalized saved to {out_eq}")

# ------------

    print("\n" + "="*60)
    print(" runtime vs events Experiment")
    print("="*60)

    plot_runtime_vs_events(results)
    out_eq = RESULTS_DIR / "runtime_vs_events.json"
    with open(out_eq, "w") as f:
        json.dump(equal_results, f, indent=2)

    print(f"\n✓ runtime vs events results saved to {out_eq}")

if __name__ == "__main__":
    main()