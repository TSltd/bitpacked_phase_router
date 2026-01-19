#!/usr/bin/env python3
"""
pbm_router_pairwise_optimized_cached.py

Optimized PBM pairwise similarity computation with caching of routed arrays.

Features:
- Self-routing once per PBM to produce feature vectors.
- Cached routed arrays stored as .npy files to avoid rerouting.
- Pairwise cosine similarity computed from cached or newly routed arrays.
- Left-alignment of rows using router.left_align_rows.
- Can handle large PBM datasets efficiently.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import router  # your pybind11 module

CACHE_DIR = "pbm_cache"

def pbm_to_array(pbm_path):
    """Load PBM image as 2D uint8 array of 0s and 1s."""
    img = Image.open(pbm_path).convert("1")
    return np.array(img, dtype=np.uint8)

def left_align_array(arr):
    """Use router's left_align_rows to left-align ones."""
    return router.left_align_rows(arr)

def run_router(S_aligned, T_aligned):
    """Run router on two left-aligned 2D arrays and return routes."""
    N, k = S_aligned.shape
    routes = np.zeros((N, k), dtype=np.int32)
    _ = router.pack_and_route(S_aligned, T_aligned, k, routes, dump=False, validate=False)
    return routes

def routes_to_feature_vector(routes):
    """Flatten routes into normalized feature vector for similarity computation."""
    vec = routes.flatten().astype(np.float32)
    max_val = np.max(vec)
    if max_val > 0:
        vec /= max_val
    return vec

def compute_pairwise_similarity(routed_arrays, names):
    """Compute pairwise cosine similarity from precomputed routed arrays."""
    N_files = len(names)
    similarity_matrix = np.zeros((N_files, N_files), dtype=np.float32)
    feature_vectors = [routes_to_feature_vector(routed_arrays[name]) for name in names]

    for i in range(N_files):
        vec_i = feature_vectors[i]
        norm_i = np.linalg.norm(vec_i) + 1e-8
        for j in range(i, N_files):
            vec_j = feature_vectors[j]
            norm_j = np.linalg.norm(vec_j) + 1e-8
            sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
            print(f"Similarity {names[i]} vs {names[j]} = {sim:.4f}")
    return similarity_matrix

def process_folder_pairwise(folder_path):
    """Load PBMs, left-align, route each once, compute pairwise similarities with caching."""
    folder = Path(folder_path)
    pbms = sorted(folder.glob("*.pbm"))
    if not pbms:
        raise ValueError("No PBM files found in folder.")

    # Ensure cache directory exists
    cache_path = folder / CACHE_DIR
    cache_path.mkdir(exist_ok=True)

    names = [pbm.name for pbm in pbms]
    routed_arrays = {}

    # Step 1: Load, left-align, route, or load from cache
    for pbm in pbms:
        cached_file = cache_path / f"{pbm.stem}_routed.npy"
        if cached_file.exists():
            routes = np.load(cached_file)
            print(f"Loaded cached routes for {pbm.name} -> shape {routes.shape}")
        else:
            arr = pbm_to_array(pbm)
            aligned = left_align_array(arr)
            routes = run_router(aligned, aligned)  # self-routing
            np.save(cached_file, routes)
            print(f"Routed {pbm.name} -> shape {routes.shape} (cached)")
        routed_arrays[pbm.name] = routes

    # Step 2: Compute pairwise similarities
    sim_matrix = compute_pairwise_similarity(routed_arrays, names)
    return names, sim_matrix

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pbm_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    names, sim_matrix = process_folder_pairwise(folder_path)

    print("\nPairwise similarity matrix:")
    print("Files:", names)
    print(sim_matrix)

if __name__ == "__main__":
    main()
