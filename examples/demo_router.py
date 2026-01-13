"""
Demo: Bit-Packed Phase Router Usage

Shows how to:
- Generate random binary matrices with NumPy / SciPy
- Pack and route them with the C++ router via Pybind11
- Integrate with PyTorch for sparse / MoE-style tensors
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from pathlib import Path
import sys

# Make sure the router module is importable
sys.path.append(str(Path(__file__).parent.parent / "src"))
import router

# ----------------------------
# 1. Generate random binary matrices
# ----------------------------
N = 1024    # rows
k = 64     # ones per row

# NumPy matrices
S_np = np.zeros((N, k), dtype=np.int32)
T_np = np.zeros((N, k), dtype=np.int32)

# Random binary matrices with exactly k ones per row
for i in range(N):
    S_np[i, :] = np.random.choice(N, size=k, replace=False)
    T_np[i, :] = np.random.choice(N, size=k, replace=False)

print("S_np:\n", S_np)
print("T_np:\n", T_np)

# ----------------------------
# 2. Pack and route using router module
# ----------------------------
routes_np = np.zeros((N, k), dtype=np.int32)

stats = router.pack_and_route(
    S_np,
    T_np,
    k,
    routes_np,
    dump=False,
    validate=True
)

print("\nRouting stats:", stats)
print("Routes array:\n", routes_np)

# ----------------------------
# 3. Convert to SciPy sparse format
# ----------------------------
S_sparse = csr_matrix((np.ones(N*k), (np.repeat(np.arange(N), k), S_np.flatten())), shape=(N, N))
T_sparse = csr_matrix((np.ones(N*k), (np.repeat(np.arange(N), k), T_np.flatten())), shape=(N, N))

print("\nS_sparse:\n", S_sparse.toarray())
print("T_sparse:\n", T_sparse.toarray())

# ----------------------------
# 4. Convert to PyTorch tensors
# ----------------------------
S_torch = torch.tensor(S_np, dtype=torch.int32)
T_torch = torch.tensor(T_np, dtype=torch.int32)
routes_torch = torch.zeros((N, k), dtype=torch.int32)

# Run router via Pybind11 with PyTorch (NumPy arrays are sufficient)
stats = router.pack_and_route(
    S_torch.numpy(),
    T_torch.numpy(),
    k,
    routes_torch.numpy(),
    dump=False,
    validate=True
)

print("\nPyTorch routes array:\n", routes_torch)
print("Routing stats:", stats)
