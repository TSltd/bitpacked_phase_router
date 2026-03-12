#!/usr/bin/env python3
"""
phase_router_expansion_test.py

Evaluate expander-like properties of the Phase Router by measuring
neighborhood expansion of row subsets.

|N(S)| / (k * |S|)

Where:
S = subset of rows
N(S) = set of columns connected to rows in S

Values close to 1.0 indicate strong expansion.

Outputs:
- CSV results
- optional plots
"""

import numpy as np
import random
import argparse
import csv
import os
import time
import matplotlib.pyplot as plt

import sys
import os


# Add project src directory to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, SRC)

from router_py import pack_and_route_np


# ---------------------------------------------------------
# Matrix generator
# ---------------------------------------------------------

def generate_binary_matrix(N, k):
    """
    Generate random binary matrices with k ones per row.
    """
    M = np.zeros((N, N), dtype=np.int32)

    for i in range(N):
        cols = np.random.choice(N, size=k, replace=False)
        M[i, cols] = 1

    return M


# ---------------------------------------------------------
# Hash router baseline
# ---------------------------------------------------------

def hash_router(N, k):
    """
    Simple hash routing baseline.
    """
    routes = np.zeros((N, k), dtype=np.int32)

    for r in range(N):
        routes[r] = np.random.choice(N, size=k, replace=False)

    return routes


# ---------------------------------------------------------
# Convert routes to adjacency list
# ---------------------------------------------------------

def routes_to_neighbors(routes):

    neighbors = []

    for r in range(routes.shape[0]):
        neighbors.append(set(routes[r]))

    return neighbors

# ---------------------------------------------------------
# Expansion test
# ---------------------------------------------------------

def expansion_test(neighbors, subset_sizes, trials):

    N = len(neighbors)
    results = []

    for s in subset_sizes:

        ratios = []

        for _ in range(trials):

            rows = random.sample(range(N), s)

            neigh = set()
            edge_count = 0

            for r in rows:
                cols = neighbors[r]
                neigh.update(cols)
                edge_count += len(cols)   # actual number of edges

            if edge_count > 0:
                ratio = len(neigh) / edge_count
            else:
                ratio = 0

            ratios.append(ratio)

        results.append({
            "subset_size": s,
            "mean": float(np.mean(ratios)),
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
            "std": float(np.std(ratios))
        })

    return results

# ---------------------------------------------------------
# Run phase router test
# ---------------------------------------------------------

def run_phase_router(N, k):

    print(f"\nRunning Phase Router expansion test N={N}, k={k}")

    S = generate_binary_matrix(N, k)
    T = generate_binary_matrix(N, k)

    start = time.time()

    routes = pack_and_route_np(S, T, k)

    runtime = time.time() - start

    neighbors = routes_to_neighbors(routes)

    return neighbors, runtime


# ---------------------------------------------------------
# Run hash baseline
# ---------------------------------------------------------

def run_hash_router(N, k):

    print(f"\nRunning Hash Router baseline N={N}, k={k}")

    routes = hash_router(N, k)

    neighbors = routes_to_neighbors(routes)

    return neighbors


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", default="test_output/expansion_results.csv")

    args = parser.parse_args()

    subset_sizes = [1, 2, 4, 8, 16, 32, 64]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Phase router
    neighbors_phase, runtime = run_phase_router(args.N, args.k)

    phase_results = expansion_test(
        neighbors_phase,
        subset_sizes,
        args.trials
    )


    # Hash baseline
    neighbors_hash = run_hash_router(args.N, args.k)

    hash_results = expansion_test(
        neighbors_hash,
        subset_sizes,
        args.trials
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    with open(args.output, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "router",
            "subset_size",
            "mean",
            "min",
            "max",
            "std"
        ])

        for r in phase_results:
            writer.writerow([
                "phase_router",
                r["subset_size"],
                r["mean"],
                r["min"],
                r["max"],
                r["std"]
            ])

        for r in hash_results:
            writer.writerow([
                "hash_router",
                r["subset_size"],
                r["mean"],
                r["min"],
                r["max"],
                r["std"]
            ])

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    print("\nExpansion Results\n")

    print("Phase Router")
    for r in phase_results:
        print(r)

    print("\nHash Router")
    for r in hash_results:
        print(r)

    # -----------------------------------------------------
    # Plot: subset size vs expansion
    # -----------------------------------------------------

    plot_dir = os.path.join(os.path.dirname(args.output), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    subset_phase = [r["subset_size"] for r in phase_results]
    expansion_phase = [r["mean"] for r in phase_results]

    subset_hash = [r["subset_size"] for r in hash_results]
    expansion_hash = [r["mean"] for r in hash_results]

    plt.figure(figsize=(8,5))

    plt.plot(subset_phase, expansion_phase, marker='o', label="Phase Router")
    plt.plot(subset_hash, expansion_hash, marker='o', label="Hash Router")

    plt.xlabel("Subset Size (|S|)")
    plt.ylabel("Expansion Ratio (unique columns / total edges)")
    plt.title("Routing Expansion vs Row Subset Size")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join(plot_dir, "expansion_vs_subset.png")
    plt.savefig(plot_path, dpi=150)

    print(f"\nSaved plot: {plot_path}")

    print(f"\nRouter runtime: {runtime:.3f} seconds")


if __name__ == "__main__":
    main()