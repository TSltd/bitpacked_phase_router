import json
import numpy as np

def load(path):
    with open(path) as f:
        return json.load(f)

def fit(data):
    X = []
    y = []

    for r in data:
        events = r["events"]
        words = r["words_touched"]
        t = r["extract_time_ms"]

        X.append([events, words, 1.0])
        y.append(t)

    X = np.array(X)
    y = np.array(y)

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


interval = load("results/density_sweep_interval.json")
original = load("results/density_sweep_original.json")

a_i, b_i, c_i = fit(interval)
a_o, b_o, c_o = fit(original)

print("\n=== INTERVAL MODEL ===")
print(f"T ≈ {a_i:.3e} * events + {b_i:.3e} * words + {c_i:.3e}")

print("\n=== ORIGINAL MODEL ===")
print(f"T ≈ {a_o:.3e} * events + {b_o:.3e} * words + {c_o:.3e}")

def evaluate(data, coeffs):
    X = []
    y = []

    for r in data:
        events = r["events"]
        words = r["words_touched"]
        t = r["extract_time_ms"]

        pred = coeffs[0]*events + coeffs[1]*words + coeffs[2]

        X.append(pred)
        y.append(t)

    X = np.array(X)
    y = np.array(y)

    error = np.mean(np.abs((X - y) / y))
    return error

print("\nInterval error:", evaluate(interval, [a_i, b_i, c_i]))
print("Original error:", evaluate(original, [a_o, b_o, c_o]))

# ------------------------------------------------------------
# HYBRID VALIDATION
# ------------------------------------------------------------

hybrid = load("results/density_sweep_hybrid.json")

interval = [r for r in hybrid if r["kernel"] == "interval"]
original = [r for r in hybrid if r["kernel"] == "original"]

def predict_interval(r):
    return a_i * r["events"] + b_i * r["words_touched"] + c_i

def predict_original(r):
    return a_o * r["events"] + b_o * r["words_touched"] + c_o

correct = 0

for r in hybrid:
    pred = "interval" if predict_interval(r) < predict_original(r) else "original"
    actual = r["kernel"]

    if pred == actual:
        correct += 1

print("\nDispatch accuracy:", correct / len(hybrid))

print("interval samples:", len(interval))
print("original samples:", len(original))

def fit_route(data):
    X = []
    y = []

    for r in data:
        N = r["N"]
        d = r["density"]
        t = r["route_time_ms"]

        X.append([
            N,
            d * N,
            d * N * N,
            np.log2(N),
            1.0
        ])
        y.append(t)

    X = np.array(X)
    y = np.array(y)

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs

ar_i, br_i, cr_i, dr_i, er_i = fit_route(interval)
ar_o, br_o, cr_o, dr_o, er_o = fit_route(original)

print("\n=== ROUTE MODEL (INTERVAL) ===")
print(f"T ≈ {ar_i:.3e}*N + {br_i:.3e}*(dN) + {cr_i:.3e}*(dN²) + {dr_i:.3e}*logN + {er_i:.3e}")

print("\n=== ROUTE MODEL (ORIGINAL) ===")
print(f"T ≈ {ar_o:.3e}*N + {br_o:.3e}*(dN) + {cr_o:.3e}*(dN²) + {dr_o:.3e}*logN + {er_o:.3e}")




def best_kernel(r):
    N = r["N"]
    d = r["density"]
    e = r["events"]
    w = r["words_touched"]

    # --- interval ---
    t_route_i = (
        ar_i * N +
        br_i * (d * N) +
        cr_i * (d * N * N) +
        dr_i * np.log2(N) +
        er_i
    )

    t_extract_i = a_i * e + b_i * w + c_i
    t_i = t_route_i + t_extract_i

    # --- original ---
    t_route_o = (
        ar_o * N +
        br_o * (d * N) +
        cr_o * (d * N * N) +
        dr_o * np.log2(N) +
        er_o
    )

    t_extract_o = a_o * e + b_o * w + c_o
    t_o = t_route_o + t_extract_o

    return "interval" if t_i < t_o else "original"

hybrid = load("results/density_sweep_hybrid.json")

correct = 0

for r in hybrid:
    pred = best_kernel(r)
    actual = r["kernel"]

    if pred == actual:
        correct += 1

print("\nFull-model dispatch accuracy:", correct / len(hybrid))

for r in hybrid[:10]:
    pred = best_kernel(r)

    t_i = a_i * r["events"] + b_i * r["words_touched"] + c_i
    t_o = a_o * r["events"] + b_o * r["words_touched"] + c_o

    print(
        f"events={r['events']:.0f}, words={r['words_touched']:.0f}, "
        f"pred={pred}, actual={r['kernel']}, "
        f"T_i={t_i:.3f}, T_o={t_o:.3f}"
    )
