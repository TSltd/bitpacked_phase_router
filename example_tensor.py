import torch
import numpy as np
import router
import time

# -------------------- Configuration --------------------
N = 8192  # Number of rows/columns
k = 64    # Number of targets per source

# -------------------- Generate random binary matrices --------------------
print(f"Generating PyTorch tensors N={N}, k={k}...")
S = torch.randint(0, 2, (N, N), dtype=torch.uint8)
T = torch.randint(0, 2, (N, N), dtype=torch.uint8)

# -------------------- Allocate routes --------------------
routes = torch.zeros((N, k), dtype=torch.int32)

# -------------------- Run unified pack-and-route --------------------
print("\nRunning pack-and-route...")
t_start = time.time()
stats = router.pack_and_route(S.numpy(), T.numpy(), k, routes.numpy())
t_end = time.time()

python_wall_time = (t_end - t_start) * 1000

# -------------------- Routing statistics --------------------
total_active = (routes != -1).sum().item()
avg_per_row = total_active / N
fill_ratio = total_active / (N * k)

# -------------------- Print results --------------------
print("\n=== Pack-and-Route Stats ===")
print(f"N = {stats['N']}, k = {stats['k']}")
print(f"Total active routes: {stats['active_routes']}")
print(f"Routes per row: {stats['routes_per_row']:.2f}")
print(f"Packing time (ms): {stats['packing_time_ms']:.2f}")
print(f"C++ routing time (ms): {stats['routing_time_ms']:.2f}")
print(f"Total time (ms): {stats['total_time_ms']:.2f}")
print(f"Python wall time (ms): {python_wall_time:.2f}")
print(f"Fill ratio: {fill_ratio:.3f}")
