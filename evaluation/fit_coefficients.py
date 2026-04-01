import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("results/density_sweep/density_sweep.json")


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def fit_model(data):
    X = []
    y = []

    for r in data:
        N = r["N"]
        f = r["fill_ratio"]

        # Total time = route + extract
        t = r.get("route_time_ms", 0.0) + r.get("extract_time_ms", 0.0)

        # Build feature vector
        events = r["events"]
        words  = r["words_touched"]

        X.append([
            1.0,
            N,                      # routing work
            r["events"],            # exact event count
            r["words_touched"],     # exact memory/scan work
        ])

        y.append(t)

    X = np.array(X)
    y = np.array(y)

    # Solve least squares
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)

    d, a, b, c = coeffs

    print("\n=== MODEL FIT ===")
    print(f"a (routing per N): {a:.6f} ms")
    print(f"b (per event):     {b:.6f} ms")
    print(f"c (scan term):     {c:.6f} ms")

    # Predictions
    pred = X @ coeffs

    rel_err = np.abs(pred - y) / np.maximum(y, 1e-9)

    print(f"\nmean relative error: {rel_err.mean():.4f}")
    print(f"median error:        {np.median(rel_err):.4f}")
    print(f"max error:           {rel_err.max():.4f}")

    return y, pred


def plot_predictions(actual, predicted):
    plt.figure()

    plt.scatter(actual, predicted, alpha=0.8)

    # Ideal line
    mn = min(actual.min(), predicted.min())
    mx = max(actual.max(), predicted.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Actual runtime (ms)")
    plt.ylabel("Predicted runtime (ms)")
    plt.title("Model Prediction vs Actual")

    plt.grid(True)

    out = "results/density_sweep/model_fit.png"
    plt.savefig(out)
    print(f"\n✓ saved plot: {out}")

def plot_predictions_dual(actual_s, pred_s, actual_l, pred_l):
    plt.figure()

    # small regime
    plt.scatter(actual_s, pred_s, label="N ≤ 8192", alpha=0.8)

    # large regime
    plt.scatter(actual_l, pred_l, label="N ≥ 16384", alpha=0.8)

    # ideal line
    all_vals = np.concatenate([actual_s, actual_l, pred_s, pred_l])
    mn = all_vals.min()
    mx = all_vals.max()
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Actual runtime (ms)")
    plt.ylabel("Predicted runtime (ms)")
    plt.title("Model Prediction (Two Regimes)")
    plt.legend()
    plt.grid(True)

    out = "results/density_sweep/model_fit_dual.png"
    plt.savefig(out)

    print(f"\n✓ saved plot: {out}")


def main():
    data = load_data()

    small = [r for r in data if r["N"] <= 8192]
    large = [r for r in data if r["N"] >= 16384]

    print("\n=== SMALL REGIME (N ≤ 8192) ===")
    actual_s, pred_s = fit_model(small)

    print("\n=== LARGE REGIME (N ≥ 16384) ===")
    actual_l, pred_l = fit_model(large)

    plot_predictions_dual(actual_s, pred_s, actual_l, pred_l)


if __name__ == "__main__":
    main()