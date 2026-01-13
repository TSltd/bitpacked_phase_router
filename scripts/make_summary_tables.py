import pandas as pd
import numpy as np

# ================================
# Load CSV
# ================================

csv_path = "evaluation_results/summary.csv"
out_path = "docs/summary_tables.md"

df = pd.read_csv(csv_path)

# Remove rows that failed (just in case)
df = df[df["cpp_total_time_ms"].notna()]

# Group by experiment configuration
group = df.groupby(["N", "k"])


# ================================
# Helper for mean ± std formatting
# ================================

def fmt_mean_std(series, unit=""):
    mean = series.mean()
    std = series.std()
    if unit:
        return f"{mean:.2f} ± {std:.2f} {unit}"
    return f"{mean:.4f} ± {std:.4f}"


# ================================
# Table 1: Performance Scaling
# ================================

perf_rows = []
for (N, k), g in group:
    perf_rows.append({
        "N": N,
        "k": k,
        "Routing time (ms)": fmt_mean_std(g["cpp_routing_time_ms"], "ms"),
        "Total time (ms)": fmt_mean_std(g["cpp_total_time_ms"], "ms"),
    })

perf_df = pd.DataFrame(perf_rows).sort_values(["N", "k"])


# ================================
# Table 2: Load Balance
# ================================

skew_rows = []
for (N, k), g in group:
    skew_rows.append({
        "N": N,
        "k": k,
        "Column mean": f"{g['col_mean'].mean():.4f}",
        "Column max": f"{g['col_max'].mean():.2f}",
        "Skew (max/mean)": fmt_mean_std(g["col_skew"]),
        "Column std": fmt_mean_std(g["col_std"]),
    })

skew_df = pd.DataFrame(skew_rows).sort_values(["N", "k"])


# ================================
# Table 3: Routing Efficiency
# ================================

eff_rows = []
for (N, k), g in group:
    eff_rows.append({
        "N": N,
        "k": k,
        "Fill ratio": fmt_mean_std(g["fill_ratio"]),
        "Coverage S": fmt_mean_std(g["coverage_S"]),
        "Coverage T": fmt_mean_std(g["coverage_T"]),
        "Active routes": fmt_mean_std(g["active_routes"]),
    })

eff_df = pd.DataFrame(eff_rows).sort_values(["N", "k"])


# ================================
# Write Markdown
# ================================

with open(out_path, "w") as f:

    f.write("# Phase Router – Summary Tables\n\n")

    f.write("## Table 1 – Routing Performance\n")
    f.write("Mean ± std over trials. Times are C++ timings.\n\n")
    f.write(perf_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Table 2 – Load Balance (Column Statistics)\n")
    f.write("Skew = max column load divided by mean column load.\n\n")
    f.write(skew_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Table 3 – Routing Efficiency\n")
    f.write("Coverage = fraction of nonzero S or T entries that were routed.\n\n")
    f.write(eff_df.to_markdown(index=False))
    f.write("\n\n")

print(f"Wrote {out_path}")
