# evaluation/plot_dispatch_boundary.py

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results/compare_original_vs_interval")

orig = json.load(open(RESULTS_DIR / "bench_original.json"))
intv = json.load(open(RESULTS_DIR / "bench_interval.json"))

orig_map = {(r["N"], r["k"]): r for r in orig}
intv_map = {(r["N"], r["k"]): r for r in intv}

# ---- YOUR CURRENT WEIGHTS ----
a = -0.323558
b = 1.915055
c = 3.565952
e = 0.452158
d = 0.0  # <-- problematic

# ---- CHANGE THIS to test ----
MARGIN = 0.1

points = []

for key in orig_map:
    if key not in intv_map:
        continue

    o = orig_map[key]
    i = intv_map[key]

    N = o["N"]
    k = o["k"]

    density = np.mean([t["density"] for t in o["trials"]])
    kn = k / N

    score = (
        a * np.log2(N) +
        b * density +
        c * kn +
        e * density * kn +
        d
    )

    t_orig = o["routing_time_ms_mean"]
    t_intv = i["routing_time_ms_mean"]

    better = t_intv < t_orig

    points.append((kn, density, score, better))

# ---- PLOT ----
plt.figure()

for kn, density, score, better in points:
    prob = 1 / (1 + np.exp(-score))
    if prob > 0.6:
        color = "blue"
    else:
        color = "red"

    marker = "o" if better else "x"

    plt.scatter(kn, density, c=color, marker=marker, s=100)

plt.xlabel("k/N")
plt.ylabel("density")

plt.title("Dispatch decision boundary\nblue=interval, red=original\ncircle=interval faster, x=original faster")

plt.grid(True)
plt.show()