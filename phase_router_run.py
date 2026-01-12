import numpy as np
import router
from pathlib import Path
import json, time, os, csv
import matplotlib.pyplot as plt
from PIL import Image

from phase_router_testing import run_routing_experiment

# ------------------------ Utilities ------------------------
def generate_random_binary_matrices(N, k_max, seed_S=None, seed_T=None):
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

def convert_pbm_to_png(pbm_files, invert=True, png_folder="dump/png"):
    png_folder = Path(png_folder)
    png_folder.mkdir(parents=True, exist_ok=True)
    png_files = []
    for pbm_file in pbm_files:
        try:
            im = Image.open(pbm_file).convert("L")
            if invert:
                im = Image.eval(im, lambda x: 255 - x)
            png_file = png_folder / (pbm_file.stem + ".png")
            im.save(png_file)
            png_files.append(png_file)
        except Exception as e:
            print(f"Failed to convert {pbm_file} to PNG: {e}")
    return png_files

def compute_route_statistics(routes, N, k):
    row_counts = np.sum(routes != -1, axis=1)
    stats = dict(
        min_routes_per_row=int(np.min(row_counts)),
        max_routes_per_row=int(np.max(row_counts)),
        mean_routes_per_row=float(np.mean(row_counts)),
        std_routes_per_row=float(np.std(row_counts))
    )
    return stats

def validate_routing(S, T, routes):
    N, k = routes.shape
    S_bits = np.sum(S)
    T_bits = np.sum(T)
    routed_bits = np.sum([S[i, routes[i] >= 0].sum() + T[i, routes[i] >= 0].sum() for i in range(N)])
    coverage = routed_bits / max(1, S_bits + T_bits)
    return coverage

# ------------------------ JSON Utilities ------------------------
def metrics_to_json_serializable(x):
    if isinstance(x, dict):
        return {k: metrics_to_json_serializable(v) for k, v in x.items()}
    if isinstance(x, list) or isinstance(x, tuple):
        return [metrics_to_json_serializable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x

# ------------------------ Main Scaling Experiment ------------------------
def run_scaling_experiment(
    N_values, k_values,
    output_root="scaling_results",
    dump_png_for_small_N=True,
    validate=True,
    max_png_N=512
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for N in N_values:
        for k in k_values:
            if k > N:
                continue
            print(f"\n=== Running N={N}, k={k} ===")
            S, T = generate_random_binary_matrices(N, k)
            routes = np.zeros((N, k), dtype=np.int32)
            run_folder = output_root / f"N_{N}_k_{k}"
            run_folder.mkdir(parents=True, exist_ok=True)
            dump_prefix = str(run_folder) if (dump_png_for_small_N and N <= 128) else None


            # ------------------------ Adaptive Multiphase Routing ------------------------
            t0 = time.time()
            active_routes = k
            png_files = []
            metrics = {}
            coverage_last = 0.0
            no_progress_count = 0
            MIN_ROUTES_PER_PHASE = max(1, k // 4)  # minimum routes required to continue
            max_phases = min(1000, 4 * k)          # limit phases relative to k

            for phase in range(1, max_phases + 1):
                phase_routes = routes[:, :active_routes]

                # Stop early if too few routes remain
                if phase_routes.shape[1] < 2:
                    print(f"Phase {phase}: too few routes ({phase_routes.shape[1]}), stopping early.")
                    break

                k_current = min(active_routes, phase_routes.shape[1])

                try:
                    m, phase_routes_update, phase_png_files = run_routing_experiment(
                        S, T, k_current,
                        dump_prefix=dump_prefix if N <= 128 else None,
                        validate=validate
                    )

                    # Update routes
                    routes[:, :k_current] = phase_routes_update
                    active_routes = k_current

                    # Only save PNGs for small N
                    if N <= 128:
                        png_files.extend(phase_png_files)

                    # Merge metrics
                    metrics.update(m)

                    # Check how many routes were actually added
                    routes_this_phase = np.sum(phase_routes_update != phase_routes)
                    if routes_this_phase < MIN_ROUTES_PER_PHASE:
                        print(f"Phase {phase}: only {routes_this_phase} routes added, below threshold {MIN_ROUTES_PER_PHASE}, stopping early.")
                        break

                    # Stop if no progress
                    if np.all(phase_routes_update == phase_routes):
                        no_progress_count += 1
                    else:
                        no_progress_count = 0

                    if no_progress_count >= 3:
                        print(f"Phase {phase}: no progress for 3 consecutive phases, stopping.")
                        break

                    # Optional: stop if coverage does not improve
                    coverage_current = metrics.get("coverage", 0.0)
                    if coverage_current <= coverage_last:
                        no_progress_count += 1
                    else:
                        no_progress_count = 0
                    coverage_last = coverage_current

                    if no_progress_count >= 3:
                        print(f"Phase {phase}: coverage stalled for 3 consecutive phases, stopping.")
                        break

                except Exception as e:
                    print(f"Phase {phase}: routing failed with k={k_current}, skipping phase. Error: {e}")
                    continue

            t1 = time.time()
            total_time = (t1 - t0) * 1000  # ms

            # Compute route statistics
            route_stats = compute_route_statistics(routes, N, k)
            active_routes_final = metrics.get("active_routes", active_routes)
            coverage = metrics.get("coverage", 0.0)
            coverage_str = f"{coverage:.3f}"

            # Merge final metrics
            metrics.update({
                "N": N,
                "k": k,
                "total_time_ms": total_time,
                **route_stats,
                "coverage": coverage,
                "active_routes": active_routes_final
            })

            summary.append(metrics)


            # ---- JSON saving ----
            metrics_json = metrics_to_json_serializable(metrics)
            with open(run_folder / "metrics.json", "w") as f:
                json.dump(metrics_json, f, indent=2)

            # ---- Save routes ----
            np.save(run_folder / "routes.npy", routes)

            # ---- Safe print ----
            print(f"Run completed: active_routes={metrics['active_routes']}, "
                  f"routing_time={metrics['total_time_ms']:.2f} ms, "
                  f"PNGs={len(png_files)}, coverage={coverage_str}")

    # ---- Save summary CSV ----
    summary_file = output_root / "summary.csv"
    keys = summary[0].keys() if summary else []
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summary:
            writer.writerow(metrics_to_json_serializable(row))

    # ---- Generate plots ----
    fig_folder = output_root / "figures"
    fig_folder.mkdir(exist_ok=True)

    plt.figure(figsize=(8,6))
    for k in k_values:
        Ns = [m['N'] for m in summary if m['k']==k]
        times = [m['total_time_ms'] for m in summary if m['k']==k]
        plt.plot(Ns, times, marker='o', label=f"k={k}")
    plt.xlabel("N (matrix size)")
    plt.ylabel("Routing time (ms)")
    plt.title("Routing time vs N for different k")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_folder / "routing_time_vs_N.png")
    plt.close()

    plt.figure(figsize=(8,6))
    for N in N_values:
        ks = [m['k'] for m in summary if m['N']==N]
        fill_ratios = [m.get('fill_ratio', 0) for m in summary if m['N']==N]
        plt.plot(ks, fill_ratios, marker='o', label=f"N={N}")
    plt.xlabel("k (max routes per row)")
    plt.ylabel("Fill ratio")
    plt.title("Fill ratio vs k for different N")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_folder / "fill_ratio_vs_k.png")
    plt.close()

    print(f"\n=== Scaling experiment completed ===")
    print(f"All results saved in {output_root}")
    return summary

# ------------------------ Entry Point ------------------------
if __name__ == "__main__":
    N_values = [256, 512, 1024, 2048]
    k_values = [8, 32, 128, 256]

    summary = run_scaling_experiment(
        N_values, k_values,
        output_root="scaling_results",
        dump_png_for_small_N=True,
        validate=True,
        max_png_N=512
    )
