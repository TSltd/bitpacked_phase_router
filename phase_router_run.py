"""
phase_router_run.py

Batch execution script for comprehensive scaling experiments.
Runs multiple test configurations and produces:
- Performance data (runtime vs N, k)
- Statistical summaries (load balance, fill ratios)
- Visual outputs (PBM -> PNG for small N)
- Reproducibility tests
- CSV/JSON exports for paper figures

This generates all evaluation data for Section 5 of the paper.
"""

import numpy as np
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from phase_router_test import (
    run_single_test,
    make_json_serializable,
    generate_random_binary_matrices
)


# ============================================================================
# Batch Experiment Runner
# ============================================================================

def run_scaling_experiment(
    N_values: List[int],
    k_values: List[int],
    output_root: str = "scaling_results",
    num_trials: int = 3,
    dump_png_max_N: int = 512,
    validate: bool = True
) -> List[Dict]:
    """
    Run comprehensive scaling experiments across multiple N and k values.
    
    Args:
        N_values: List of matrix sizes to test
        k_values: List of k values to test
        output_root: Output directory for results
        num_trials: Number of trials per configuration (for averaging)
        dump_png_max_N: Maximum N for PBM->PNG dumps (small N only)
        validate: Whether to validate routes
    
    Returns:
        List of all metrics dictionaries
    """
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    print(f"\n{'='*70}")
    print(f"SCALING EXPERIMENT")
    print(f"N values: {N_values}")
    print(f"k values: {k_values}")
    print(f"Trials per config: {num_trials}")
    print(f"Output: {output_root}")
    print(f"{'='*70}\n")
    
    total_tests = len(N_values) * len(k_values) * num_trials
    test_num = 0
    
    for N in N_values:
        for k in k_values:
            if k > N:
                print(f"Skipping N={N}, k={k} (k > N)")
                continue
            
            trial_results = []
            
            for trial in range(num_trials):
                test_num += 1
                print(f"\n[Test {test_num}/{total_tests}] N={N}, k={k}, trial={trial+1}/{num_trials}")
                
                # Create output folder for this run
                run_folder = output_path / f"N{N}_k{k}_trial{trial}"
                run_folder.mkdir(parents=True, exist_ok=True)
                
                # Determine if we should dump PBMs
                dump_prefix = str(run_folder) if N <= dump_png_max_N else None
                
                # Run the test
                try:
                    metrics = run_single_test(
                        N=N,
                        k=k,
                        seed_S=None,  # Random for each trial
                        seed_T=None,
                        dump_prefix=dump_prefix,
                        validate=validate
                    )
                    
                    metrics["trial"] = trial
                    trial_results.append(metrics)
                    all_results.append(metrics)
                    
                    # Save individual run metrics
                    with open(run_folder / "metrics.json", "w") as f:
                        json.dump(make_json_serializable(metrics), f, indent=2)
                    
                except Exception as e:
                    print(f"ERROR in test N={N}, k={k}, trial={trial}: {e}")
                    continue
            
            # Compute aggregate statistics across trials
            if trial_results:
                aggregate = compute_aggregate_metrics(trial_results)
                aggregate_path = output_path / f"aggregate_N{N}_k{k}.json"
                with open(aggregate_path, "w") as f:
                    json.dump(make_json_serializable(aggregate), f, indent=2)
                
                print(f"\nAggregate for N={N}, k={k}:")
                print(f"  Routing time: {aggregate['routing_time_ms_mean']:.2f} ± {aggregate['routing_time_ms_std']:.2f} ms")
                print(f"  Fill ratio: {aggregate['fill_ratio_mean']:.3f} ± {aggregate['fill_ratio_std']:.3f}")
                print(f"  Column skew: {aggregate['col_skew_mean']:.2f} ± {aggregate['col_skew_std']:.2f}")
    
    print(f"\n{'='*70}")
    print(f"SCALING EXPERIMENT COMPLETED")
    print(f"Total tests run: {len(all_results)}")
    print(f"{'='*70}\n")
    
    return all_results


def compute_aggregate_metrics(trial_results: List[Dict]) -> Dict:
    """
    Compute mean and std across multiple trials.
    
    Args:
        trial_results: List of metrics dicts from multiple trials
    
    Returns:
        Dictionary with mean and std for each metric
    """
    if not trial_results:
        return {}
    
    # Identify numeric keys
    numeric_keys = []
    for key, value in trial_results[0].items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_keys.append(key)
    
    aggregate = {}
    for key in numeric_keys:
        values = [r[key] for r in trial_results if key in r]
        if values:
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))
            aggregate[f"{key}_min"] = float(np.min(values))
            aggregate[f"{key}_max"] = float(np.max(values))
    
    # Add config info
    aggregate["N"] = trial_results[0]["N"]
    aggregate["k"] = trial_results[0]["k"]
    aggregate["num_trials"] = len(trial_results)
    
    return aggregate


# ============================================================================
# Reproducibility Test
# ============================================================================

def run_reproducibility_test(
    N: int = 1024,
    k: int = 64,
    num_runs: int = 5,
    output_root: str = "reproducibility_test"
) -> Dict:
    """
    Test reproducibility with fixed seeds.
    All runs should produce identical route arrays.
    
    Args:
        N: Matrix size
        k: Max routes per row
        num_runs: Number of repeated runs
        output_root: Output directory
    
    Returns:
        Dictionary with reproducibility test results
    """
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"REPRODUCIBILITY TEST")
    print(f"N={N}, k={k}, runs={num_runs}")
    print(f"Using fixed seeds (S=42, T=123)")
    print(f"{'='*70}\n")
    
    # Use fixed seeds for all runs
    seed_S = 42
    seed_T = 123
    
    all_routes = []
    all_metrics = []
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}...")
        
        # Generate with fixed seeds
        from phase_router_test import generate_random_binary_matrices
        import router
        
        S, T = generate_random_binary_matrices(N, k, seed_S, seed_T)
        routes = np.zeros((N, k), dtype=np.int32)
        
        stats = router.pack_and_route(S, T, k, routes, dump=False, validate=False)
        
        all_routes.append(routes.copy())
        all_metrics.append(stats)
    
    # Check if all routes are identical
    reference = all_routes[0]
    all_identical = all(np.array_equal(reference, r) for r in all_routes)
    
    result = {
        "N": N,
        "k": k,
        "num_runs": num_runs,
        "seed_S": seed_S,
        "seed_T": seed_T,
        "all_identical": all_identical,
        "test_passed": all_identical
    }
    
    if all_identical:
        print(f"✓ REPRODUCIBILITY TEST PASSED")
        print(f"  All {num_runs} runs produced identical routes")
    else:
        print(f"✗ REPRODUCIBILITY TEST FAILED")
        print(f"  Routes differ between runs!")
        
        # Report differences
        for i in range(1, num_runs):
            diff = np.sum(all_routes[i] != reference)
            result[f"diff_run{i}"] = int(diff)
            print(f"  Run {i+1} differs by {diff} entries")
    
    # Save result
    with open(output_path / "reproducibility_result.json", "w") as f:
        json.dump(make_json_serializable(result), f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return result


# ============================================================================
# Generate Summary Reports
# ============================================================================

def generate_summary_csv(results: List[Dict], output_path: Path):
    """
    Generate CSV summary of all results.
    
    Args:
        results: List of all metrics dictionaries
        output_path: Output file path
    """
    if not results:
        print("No results to summarize")
        return
    
    # Collect all keys
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    # Remove nested dicts
    csv_keys = [k for k in all_keys if not isinstance(results[0].get(k), dict)]
    csv_keys.sort()
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        
        for result in results:
            row = {k: result.get(k, "") for k in csv_keys}
            writer.writerow(make_json_serializable(row))
    
    print(f"CSV summary saved: {output_path}")


def generate_plots(results: List[Dict], output_folder: Path):
    """
    Generate performance and scaling plots.
    
    Args:
        results: List of all metrics dictionaries
        output_folder: Output directory for plots
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    data = {}
    for r in results:
        N = r["N"]
        k = r["k"]
        if (N, k) not in data:
            data[(N, k)] = []
        data[(N, k)].append(r)
    
    # Plot 1: Routing time vs N (for different k)
    plt.figure(figsize=(10, 6))
    k_values = sorted(set(r["k"] for r in results))
    
    for k in k_values:
        N_vals = []
        time_means = []
        time_stds = []
        
        for (N, k_val), metrics in sorted(data.items()):
            if k_val == k:
                times = [m["routing_time_ms"] for m in metrics]
                N_vals.append(N)
                time_means.append(np.mean(times))
                time_stds.append(np.std(times))
        
        plt.errorbar(N_vals, time_means, yerr=time_stds, marker='o', label=f"k={k}", capsize=5)
    
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Routing Time (ms)")
    plt.title("Routing Performance vs Matrix Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "routing_time_vs_N.png", dpi=150)
    plt.close()
    print(f"Saved: {output_folder / 'routing_time_vs_N.png'}")
    
    # Plot 2: Column load balance (skew) vs N
    plt.figure(figsize=(10, 6))
    
    for k in k_values:
        N_vals = []
        skew_means = []
        skew_stds = []
        
        for (N, k_val), metrics in sorted(data.items()):
            if k_val == k:
                skews = [m["col_skew"] for m in metrics]
                N_vals.append(N)
                skew_means.append(np.mean(skews))
                skew_stds.append(np.std(skews))
        
        plt.errorbar(N_vals, skew_means, yerr=skew_stds, marker='s', label=f"k={k}", capsize=5)
    
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Column Skew (max/mean)")
    plt.title("Load Balance vs Matrix Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=2.0, color='r', linestyle='--', label='2x threshold', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_folder / "column_skew_vs_N.png", dpi=150)
    plt.close()
    print(f"Saved: {output_folder / 'column_skew_vs_N.png'}")
    
    # Plot 3: Fill ratio vs k
    plt.figure(figsize=(10, 6))
    N_values = sorted(set(r["N"] for r in results))
    
    for N in N_values:
        k_vals = []
        fill_means = []
        fill_stds = []
        
        for (N_val, k), metrics in sorted(data.items()):
            if N_val == N:
                fills = [m["fill_ratio"] for m in metrics]
                k_vals.append(k)
                fill_means.append(np.mean(fills))
                fill_stds.append(np.std(fills))
        
        plt.errorbar(k_vals, fill_means, yerr=fill_stds, marker='d', label=f"N={N}", capsize=5)
    
    plt.xlabel("Max Routes per Row (k)")
    plt.ylabel("Fill Ratio")
    plt.title("Routing Fill Ratio vs k")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "fill_ratio_vs_k.png", dpi=150)
    plt.close()
    print(f"Saved: {output_folder / 'fill_ratio_vs_k.png'}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main execution function for comprehensive evaluation.
    """
    print("\n" + "="*70)
    print(" BIT-PACKED PHASE ROUTER - COMPREHENSIVE EVALUATION SUITE")
    print("="*70 + "\n")
    
    # Configuration
    N_values = [256, 512, 1024, 2048, 4096]
    k_values = [8, 16, 64, 256, 512]
    num_trials = 3
    output_root = "evaluation_results"
    
    # Run scaling experiments
    print("\n[1/3] Running scaling experiments...")
    results = run_scaling_experiment(
        N_values=N_values,
        k_values=k_values,
        output_root=output_root,
        num_trials=num_trials,
        dump_png_max_N=512,
        validate=True
    )
    
    # Generate summaries
    print("\n[2/3] Generating summary reports...")
    output_path = Path(output_root)
    generate_summary_csv(results, output_path / "summary.csv")
    generate_plots(results, output_path / "figures")
    
    # Run reproducibility test
    print("\n[3/3] Running reproducibility test...")
    repro_result = run_reproducibility_test(
        N=1024,
        k=64,
        num_runs=5,
        output_root=str(output_path / "reproducibility")
    )
    
    # Final summary
    print("\n" + "="*70)
    print(" EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_root}/")
    print(f"  - summary.csv: All metrics in CSV format")
    print(f"  - figures/: Performance plots")
    print(f"  - reproducibility/: Reproducibility test results")
    print(f"  - N*_k*_trial*/: Individual test outputs")
    print(f"\nTotal tests: {len(results)}")
    print(f"Reproducibility: {'PASSED' if repro_result['test_passed'] else 'FAILED'}")
    print("\n")


if __name__ == "__main__":
    main()
