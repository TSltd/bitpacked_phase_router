#!/usr/bin/env python3
"""
pbm_router_folder.py

Compute routing-based feature vectors for a folder of PBM images
and optionally produce a pairwise similarity matrix.
The feature vectors encode the row-wise structure and overlap patterns of ones.
The similarity matrix can be used for clustering or comparing images.

Usage:
python pbm_router_folder.py ./pbm_images/

Prints stats for each PBM
Prints a pairwise cosine similarity matrix between routed features
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import router  # your pybind11 module

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
    stats = router.pack_and_route(S_aligned, T_aligned, k, routes, dump=False, validate=False)
    return routes, stats

def routes_to_feature_vector(routes):
    """Flatten routes into normalized feature vector for ML."""
    vec = routes.flatten().astype(np.float32)
    max_val = np.max(vec)
    if max_val > 0:
        vec /= max_val
    return vec

def process_folder(folder_path):
    """Process all PBM images in a folder and return features dict."""
    folder = Path(folder_path)
    pbms = sorted(folder.glob("*.pbm"))
    if not pbms:
        raise ValueError("No PBM files found in folder.")

    features = {}
    for pbm in pbms:
        arr = pbm_to_array(pbm)
        arr_aligned = left_align_array(arr)
        # For self-routing, just route array against itself
        routes, stats = run_router(arr_aligned, arr_aligned)
        feature_vector = routes_to_feature_vector(routes)
        features[pbm.name] = feature_vector
        print(f"Processed {pbm.name}: {stats}")

    return features

def compute_pairwise_similarity(features):
    """Compute cosine similarity matrix between feature vectors."""
    names = list(features.keys())
    vecs = np.stack([features[n] for n in names])
    # Normalize vectors
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_normed = vecs / (norms + 1e-8)
    sim_matrix = vecs_normed @ vecs_normed.T
    return names, sim_matrix

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pbm_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    features = process_folder(folder_path)
    names, sim_matrix = compute_pairwise_similarity(features)

    print("\nPairwise similarity matrix:")
    print("Files:", names)
    print(sim_matrix)

if __name__ == "__main__":
    main()
