#!/usr/bin/env python3
"""
pbm_router_feature.py

Standalone script to compute a routing-based feature vector from two PBM images.
Prints routing stats and a normalized feature vector.
The vector can be used directly in ML pipelines: 
similarity models, contrastive learning, clustering, etc.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import router 

def pbm_to_array(pbm_path):
    """Load PBM image and return as 2D uint8 array of 0s and 1s."""
    img = Image.open(pbm_path).convert("1")  # convert to 1-bit black/white
    return np.array(img, dtype=np.uint8)

def left_align_array(arr):
    """Use the router's left_align_rows to align 1s left per row."""
    return router.left_align_rows(arr)

def run_router(S_aligned, T_aligned):
    """Run the router on two left-aligned 2D arrays and return the routes array."""
    N, k = S_aligned.shape
    routes = np.zeros((N, k), dtype=np.int32)
    stats = router.pack_and_route(S_aligned, T_aligned, k, routes, dump=False, validate=False)
    return routes, stats

def routes_to_feature_vector(routes):
    """Flatten routes into a normalized feature vector suitable for ML."""
    vec = routes.flatten().astype(np.float32)
    max_val = np.max(vec)
    if max_val > 0:
        vec /= max_val  # normalize to 0-1
    return vec

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <S.pbm> <T.pbm>")
        sys.exit(1)

    S_path = Path(sys.argv[1])
    T_path = Path(sys.argv[2])

    # Load PBMs
    S_array = pbm_to_array(S_path)
    T_array = pbm_to_array(T_path)

    # Ensure both are square and same size
    if S_array.shape != T_array.shape:
        print("Error: PBMs must have the same dimensions.")
        sys.exit(1)

    # Left-align
    S_aligned = left_align_array(S_array)
    T_aligned = left_align_array(T_array)

    # Run router
    routes, stats = run_router(S_aligned, T_aligned)

    # Convert to feature vector
    feature_vector = routes_to_feature_vector(routes)

    # Output info
    print(f"Routing stats: {stats}")
    print(f"Feature vector (shape {feature_vector.shape}):\n{feature_vector}")

if __name__ == "__main__":
    main()
