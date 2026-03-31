import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/compare_original_vs_interval")

orig = json.load(open(RESULTS_DIR / "bench_original.json"))
intv = json.load(open(RESULTS_DIR / "bench_interval.json"))

orig_map = {(r["N"], r["k"]): r for r in orig}
intv_map = {(r["N"], r["k"]): r for r in intv}

X = []
y = []

MARGIN = 1.10  # require 10% speedup to choose interval

for key in sorted(orig_map.keys()):
    if key not in intv_map:
        continue

    o = orig_map[key]
    i = intv_map[key]

    N = o["N"]
    k = o["k"]

    density = np.mean([t["density"] for t in o["trials"]])
    kn = k / N

    t_orig = o["routing_time_ms_mean"]
    t_intv = i["routing_time_ms_mean"]

    # LABEL WITH MARGIN
    if t_intv < t_orig / MARGIN:
        label = 1  # interval
    else:
        label = 0  # original (safe fallback)

    X.append([
        np.log2(N),
        density,
        kn,
        density * kn,   # interaction term
        1.0
    ])
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset size:", len(X))
print("Interval chosen:", np.sum(y))

# ------------------------------------------------------------
# Logistic regression
# ------------------------------------------------------------

w = np.zeros(X.shape[1])
lr = 0.1

for epoch in range(3000):
    logits = X @ w
    probs = 1 / (1 + np.exp(-logits))

    grad = X.T @ (probs - y) / len(X)
    w -= lr * grad

    if epoch % 300 == 0:
        loss = -np.mean(y * np.log(probs + 1e-9) +
                        (1 - y) * np.log(1 - probs + 1e-9))
        print(f"epoch {epoch:4d} loss={loss:.4f}")

a, b, c, e, d = w

print("\n=== WEIGHTS (SAFE / MARGIN-AWARE) ===")
print(f"a (logN)     = {a:.6f}")
print(f"b (density)  = {b:.6f}")
print(f"c (k/N)      = {c:.6f}")
print(f"d (bias)     = {d:.6f}")
print(f"e (d*kn)     = {e:.6f}")

# ------------------------------------------------------------
# Evaluate regret
# ------------------------------------------------------------

regret = 0
for i in range(len(X)):
    score = X[i] @ w
    pred = 1 if score > 0 else 0

    actual = y[i]

    if pred != actual:
        regret += 1

print(f"\nMisclassifications: {regret}/{len(X)}")

# stricter: check slowdowns
slowdowns = 0
for key in orig_map:
    if key not in intv_map:
        continue

    o = orig_map[key]
    i = intv_map[key]

    # --- DEBUG: speedup stats ---
    speedups = []
    for key in orig_map:
        if key in intv_map:
            o = orig_map[key]
            i = intv_map[key]
            speedups.append(o["routing_time_ms_mean"] / i["routing_time_ms_mean"])

    print("min speedup:", min(speedups))
    print("max speedup:", max(speedups))
    print("mean speedup:", sum(speedups) / len(speedups))

    N = o["N"]
    k = o["k"]
    density = np.mean([t["density"] for t in o["trials"]])

    kn = k / N
    x = np.array([np.log2(N), density, kn, density * kn, 1.0])
    pred = (x @ w > 0)

    t_orig = o["routing_time_ms_mean"]
    t_intv = i["routing_time_ms_mean"]

    if pred and t_intv > t_orig:
        slowdowns += 1

print(f"Actual slowdowns: {slowdowns}")