# router_py.py
"""
High-level Python wrapper for Bit-Packed Phase Router (pybind11 C++ module).

Provides interfaces for:
- NumPy dense arrays
- SciPy sparse matrices (CSR)
- PyTorch tensors
- Optional conversion to index-based routing for MoE layers
"""

from typing import Optional
import numpy as np

import router  # pybind11 module

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.sparse import csr_matrix, isspmatrix_csr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# -----------------------------
# NumPy Interface
# -----------------------------

def pack_and_route_np(S: np.ndarray, T: np.ndarray, k: int, validate: bool = False) -> np.ndarray:
    """
    Run routing on NumPy dense arrays.
    
    Args:
        S: (N, k) binary matrix
        T: (N, k) binary matrix
        k: max routes per row
        validate: whether to validate routing

    Returns:
        routes: (N, k) array of routed indices
    """
    assert S.shape == T.shape, "S and T must have same shape"
    routes = np.zeros_like(S, dtype=np.int32)
    router.pack_and_route(S, T, k, routes, validate=validate)
    return routes


# -----------------------------
# SciPy Sparse Interface
# -----------------------------

def pack_and_route_sparse(S: 'csr_matrix', T: 'csr_matrix', k: int, validate: bool = False) -> np.ndarray:
    """
    Run routing on SciPy CSR sparse matrices.

    Args:
        S, T: CSR matrices of shape (N, k)
        k: max routes per row
        validate: whether to validate routing

    Returns:
        routes: dense NumPy array (N, k)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required for sparse interface")
    if not (isspmatrix_csr(S) and isspmatrix_csr(T)):
        raise TypeError("S and T must be CSR matrices")
    S_dense = S.toarray().astype(np.int32)
    T_dense = T.toarray().astype(np.int32)
    return pack_and_route_np(S_dense, T_dense, k, validate)


# -----------------------------
# PyTorch Interface
# -----------------------------

def pack_and_route_torch(S: 'torch.Tensor', T: 'torch.Tensor', k: int, validate: bool = False) -> 'torch.Tensor':
    """
    Run routing on PyTorch CPU tensors.

    Args:
        S, T: (N, k) int32 tensors
        k: max routes per row
        validate: whether to validate routing

    Returns:
        routes: (N, k) int32 tensor
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this interface")
    assert S.shape == T.shape, "S and T must have same shape"
    assert S.device.type == 'cpu' and T.device.type == 'cpu', "Only CPU tensors supported currently"
    
    S_np = S.numpy().astype(np.int32)
    T_np = T.numpy().astype(np.int32)
    routes_np = pack_and_route_np(S_np, T_np, k, validate)
    return torch.from_numpy(routes_np)


# -----------------------------
# Optional: Index Conversion
# -----------------------------

def routes_to_indices(routes: np.ndarray) -> np.ndarray:
    """
    Convert binary route matrix (N, k) -> indices of active routes per row.
    
    Args:
        routes: binary (0/1) matrix, shape (N, k)
    
    Returns:
        indices: (N, <=k) array with actual routed positions
    """
    idx_list = [np.flatnonzero(row) for row in routes]
    max_len = max(len(r) for r in idx_list)
    indices = np.full((routes.shape[0], max_len), -1, dtype=np.int32)
    for i, r in enumerate(idx_list):
        indices[i, :len(r)] = r
    return indices
