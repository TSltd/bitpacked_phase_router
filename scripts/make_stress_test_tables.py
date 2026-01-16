import json
from pathlib import Path

json_path = Path("test_output/stress_test_results.json")
md_path   = Path("test_output/stress_test.md")

with open(json_path) as f:
    data = json.load(f)

lines = []

# -------------------------------
# Single-Phase Table
# -------------------------------
lines.append("## Single-Phase Stress Sweep\n")
lines.append("| k | Phase Router Time (ms) | Active Routes | Routes/Row | Column Skew | Fill Ratio | Hash Router Time (ms) | Hash Column Skew |")
lines.append("|---|-----------------------|---------------|------------|-------------|------------|----------------------|-----------------|")

for entry in data["single_phase"]:
    k = entry["k"]
    phase_time = entry["phase_router"]["time_ms"]
    active_routes = entry["phase_router"]["fill"]["active_routes"]
    routes_per_row = entry["phase_router"]["fill"]["routes_per_row"]
    col_skew = entry["phase_router"]["stats"]["col_skew"]
    fill_ratio = entry["phase_router"]["fill"]["fill_ratio"]
    
    hash_time = entry["hash_router"]["time_ms"]
    hash_skew = entry["hash_router"]["stats"]["col_skew"]

    lines.append(f"| {k} | {phase_time:.1f} | {active_routes} | {routes_per_row:.4f} | {col_skew:.2f} | {fill_ratio:.5f} | {hash_time:.1f} | {hash_skew:.2f} |")

# -------------------------------
# Two-Phase Adversarial Table
# -------------------------------
lines.append("\n## Two-Phase Adversarial Test\n")
lines.append("| k | Phase Router Phase 1 Time (ms) | Phase Router Phase 2 Time (ms) | Phase 2 Max Column Load | Phase 2 Column Skew | Hash Router Phase 1 Time (ms) | Hash Router Phase 2 Time (ms) | Hash Phase 2 Max Column Load | Hash Phase 2 Column Skew |")
lines.append("|---|-------------------------------|-------------------------------|------------------------|------------------|-------------------------------|-------------------------------|---------------------------|-------------------------|")

for entry in data["two_phase_adversarial"]:
    k = entry["k"]

    phase1_time = entry["phase_router"]["time_ms"]["phase1"]
    phase2_time = entry["phase_router"]["time_ms"]["phase2"]
    phase2_stats = entry["phase_router"]["phase2"]
    phase2_max = phase2_stats["col_max"]
    phase2_skew = phase2_stats["col_skew"]

    hash1_time = entry["hash_router"]["time_ms"]["phase1"]
    hash2_time = entry["hash_router"]["time_ms"]["phase2"]
    hash2_stats = entry["hash_router"]["phase2"]
    hash2_max = hash2_stats["col_max"]
    hash2_skew = hash2_stats["col_skew"]

    lines.append(f"| {k} | {phase1_time:.1f} | {phase2_time:.1f} | {phase2_max} | {phase2_skew:.2f} | {hash1_time:.1f} | {hash2_time:.1f} | {hash2_max} | {hash2_skew:.2f} |")

# -------------------------------
# Image placeholders
# -------------------------------
lines.append("\n### Plots\n")
for k in [64, 256, 1024]:
    lines.append(f"![Phase 2 Load k={k}](img/phase2_load_N32000_k{k}.png)")

lines.append("![Phase 2 Routing Time](img/phase1_plus_phase2_routing_time.png)")

# -------------------------------
# Write to file
# -------------------------------
with open(md_path, "w") as f:
    f.write("\n".join(lines))

print(f"Markdown report written to {md_path}")
