import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/density_sweep")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_json(name):
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"Missing: {path}")
        return []
    with open(path) as f:
        return json.load(f)


density_data = load_json("density_sweep.json")
equal_data   = load_json("equal_work.json")


# ------------------------------------------------------------
# 1. Density sweep analysis
# ------------------------------------------------------------

def plot_density_sweep(data):
    print("\n=== DENSITY SWEEP SUMMARY ===")

    # group by N
    by_N = {}
    for r in data:
        by_N.setdefault(r["N"], []).append(r)

    plt.figure()

    for N, rows in sorted(by_N.items()):
        rows = sorted(rows, key=lambda x: x["density"])

        densities = [r["density"] for r in rows]
        times     = [r["routing_time_ms_mean"] for r in rows]

        print(f"\nN = {N}")
        for r in rows:
            print(f"  d={r['density']:.4f} → {r['routing_time_ms_mean']:.2f} ms")

        plt.plot(densities, times, marker="o", label=f"N={N}")

    plt.xlabel("Input density")
    plt.ylabel("Routing time (ms)")
    plt.title("Runtime vs Density")
    plt.legend()
    plt.grid()

    plt.xscale("log")
    plt.yscale("log")

    out = RESULTS_DIR / "density_vs_runtime.png"
    plt.savefig(out)
    print(f"\n✓ saved plot: {out}")


# ------------------------------------------------------------
# 2. Equal-work analysis
# ------------------------------------------------------------

def plot_equal_work(data):
    print("\n=== EQUAL WORK SUMMARY ===")

    # group by target
    by_target = {}
    for r in data:
        key = r.get("target_density_product", None)
        by_target.setdefault(key, []).append(r)

    plt.figure()

    for target, rows in sorted(by_target.items()):
        rows = sorted(rows, key=lambda x: x["N"])

        Ns    = [r["N"] for r in rows]
        times = [r["routing_time_ms"] for r in rows]

        print(f"\nTarget ≈ {target}")
        for r in rows:
            print(f"  N={r['N']} → {r['routing_time_ms']:.2f} ms")

        plt.plot(Ns, times, marker="o", label=f"target={target}")

    plt.xlabel("N")
    plt.ylabel("Routing time (ms)")
    plt.title("Equal Work: Runtime vs N")
    plt.legend()
    plt.grid()

    plt.xscale("log")
    plt.yscale("log")

    out = RESULTS_DIR / "equal_work.png"
    plt.savefig(out)
    print(f"\n✓ saved plot: {out}")

def plot_runtime_vs_output_size(data):
    import matplotlib.pyplot as plt
    from collections import defaultdict

    grouped = defaultdict(list)

    # group by N
    for r in data:
        grouped[r["N"]].append(r)

    plt.figure()

    for N, results in grouped.items():
        fills = [r["fill_ratio"] for r in results]
        times = [r["routing_time_ms_mean"] for r in results]

        plt.plot(fills, times, marker="o", label=f"N={N}")

    plt.xlabel("Output density (fill ratio)")
    plt.ylabel("Routing time (ms)")
    plt.title("Runtime vs Output Size")
    plt.legend()
    plt.grid(True)

    plt.xscale("log")
    plt.yscale("log")

    out = RESULTS_DIR / "runtime_vs_output.png"
    plt.savefig(out)
    print(f"\n✓ saved plot: {out}")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

def main():
    if density_data:
        plot_density_sweep(density_data)
        plot_runtime_vs_output_size(density_data) 

    if equal_data:
        plot_equal_work(equal_data)


if __name__ == "__main__":
    main()