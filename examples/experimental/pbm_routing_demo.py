#!/usr/bin/env python3
"""
Sparse PBM router runner (direct P4 output)
This script turns black-and-white images (PBMs) into structured binary matrices, 
aligns them, and routes them efficiently

Usage:
    python pbm_router_sparse_direct.py S.pbm T.pbm output_routes.pbm
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.sparse import csr_matrix
import router_py as router_module 

def load_pbm_sparse(path: str) -> csr_matrix:
    """Load a PBM image as a sparse CSR matrix of 0/1"""
    img = Image.open(path).convert('1')  # 1-bit mode
    arr = np.array(img, dtype=np.uint8)
    row_idx, col_idx = np.nonzero(arr)
    data = np.ones(len(row_idx), dtype=np.uint8)
    return csr_matrix((data, (row_idx, col_idx)), shape=arr.shape)

def save_sparse_as_pbm_p4(matrix: csr_matrix, path: str):
    """Save sparse 0/1 CSR matrix as binary PBM (P4)"""
    N, M = matrix.shape
    row_bytes = (M + 7) // 8
    with open(path, 'wb') as f:
        f.write(f"P4\n{M} {N}\n".encode())
        for i in range(N):
            row_data = np.zeros(row_bytes, dtype=np.uint8)
            cols = matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]
            for col in cols:
                row_data[col // 8] |= 0x80 >> (col % 8)
            f.write(row_data.tobytes())

def main():
    if len(sys.argv) < 4:
        print("Usage: python pbm_router_sparse_direct.py S.pbm T.pbm output_routes.pbm")
        sys.exit(1)

    S_path = sys.argv[1]
    T_path = sys.argv[2]
    out_path = sys.argv[3]

    # -------------------
    # 1. Load PBMs as sparse matrices
    # -------------------
    S_sparse = load_pbm_sparse(S_path)
    T_sparse = load_pbm_sparse(T_path)

    if S_sparse.shape != T_sparse.shape:
        raise ValueError(f"S and T must have same shape, got {S_sparse.shape} vs {T_sparse.shape}")

    N = S_sparse.shape[0]
    k = min(S_sparse.max(), T_sparse.max())

    # -------------------
    # 2. Run router on sparse matrices
    # -------------------
    routes = router_module.pack_and_route_sparse(S_sparse, T_sparse, k, validate=True)
    print("Routes array (indices, -1 means unused):\n", routes)

    # -------------------
    # 3. Build sparse routed matrix directly
    # -------------------
    row_idx = []
    col_idx = []
    for i in range(N):
        for idx in routes[i]:
            if idx >= 0 and idx < N:
                row_idx.append(i)
                col_idx.append(idx)
    data = np.ones(len(row_idx), dtype=np.uint8)
    routed_sparse = csr_matrix((data, (row_idx, col_idx)), shape=(N, N))

    # -------------------
    # 4. Save directly as binary PBM (P4)
    # -------------------
    save_sparse_as_pbm_p4(routed_sparse, out_path)
    print(f"Routed PBM saved to {out_path}")

if __name__ == "__main__":
    main()
