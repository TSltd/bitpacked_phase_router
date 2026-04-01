import json
import numpy as np

data = json.load(open("results/density_sweep_hybrid.json"))

# ------------------------------------------------------------
# Evaluate threshold on k/N
# ------------------------------------------------------------

def evaluate(threshold):
    correct = 0
    for r in data:
        kn = r["k"] / r["N"]
        pred = "original" if kn > threshold else "interval"
        if pred == r["kernel"]:
            correct += 1
    return correct / len(data)

# Sweep thresholds
thresholds = np.linspace(0.01, 1.0, 200)

best_t = None
best_acc = 0

for t in thresholds:
    acc = evaluate(t)
    if acc > best_acc:
        best_acc = acc
        best_t = t

print("\n=== BEST FLOAT THRESHOLD ===")
print("threshold (k/N):", best_t)
print("accuracy:", best_acc)


# ------------------------------------------------------------
# Convert to integer rule: k > N / divisor
# ------------------------------------------------------------

divisors = [2, 3, 4, 5, 6, 8, 10]

print("\n=== INTEGER RULES ===")

best_div = None
best_div_acc = 0

for d in divisors:
    correct = 0
    for r in data:
        pred = "original" if r["k"] > (r["N"] / d) else "interval"
        if pred == r["kernel"]:
            correct += 1
    acc = correct / len(data)

    print(f"k > N/{d}  -> accuracy = {acc:.3f}")

    if acc > best_div_acc:
        best_div_acc = acc
        best_div = d

print("\n=== BEST INTEGER RULE ===")
print(f"k > N/{best_div}")
print("accuracy:", best_div_acc)

for r in data:
    if r["kernel"] == "original":
        print(r["k"] / r["N"])

print(sorted(set(r["k"] for r in data if r["kernel"] == "original")))