## ** Phase Router vs Hash Evaluation**

This script performs a **comprehensive stress test** comparing the bit-packed **phase router** against a simple **hash-based router**. It includes:

- Single-phase routing tests (various `k` values)
- Two-phase **adversarial composability tests** where hash routing collapses
- Automatic generation of:

  - Column load histograms (per router)
  - Routing statistics (skew, min/max, fill ratios)
  - JSON, CSV, and Markdown results tables
  - Optional plots (`--skip-plots` to disable for memory savings)

- Full reproducibility across machines and seeds
- Markdown tables including system specs (CPU, threads, RAM) for easy benchmarking

### **Run the Script**

From the project root:

```bash
# Standard run (plots enabled)
python evaluation/phase_router_vs_hash.py

# Skip plotting to save memory and speed up runs
python evaluation/phase_router_vs_hash.py --skip-plots
```

> Output files are saved under `test_output/`:
>
> - `single_phase_results.csv`
> - `two_phase_adversarial_results.csv`
> - `two_phase_adversarial_results.md`
> - Optional PNG plots in `test_output/plots/`

### **Adjust Test Parameters**

Inside the script (`phase_router_vs_hash.py`):

- `N` → number of rows/nodes (default: 16,000)
- `ks_single` → list of k values for single-phase tests
- `ks_two` → list of k values for two-phase adversarial tests

Example snippet:

```python
N = 16000
ks_single = [16, 64, 256, 1024, 4096, 12800]
ks_two = [16, 64, 256, 1024, 4096, 12800]
```

### **Requirements**

- All core dependencies from `requirements.txt`
- Optional but recommended: `matplotlib` and `psutil` for plots and system info

```bash
pip install -r requirements.txt
```

### **Quick Tips**

- Large `N` or high `k` can consume significant memory. Use `--skip-plots` to reduce footprint.
- CSV/Markdown outputs are fully reproducible; system specs are appended to Markdown for easy benchmarking.
- Use the results for capacity planning, skew analysis, and MoE expert allocation studies.

---

### **Quick Example**

Run the script for a single `k` value and print summary statistics:

```bash
# Run the script (plots optional)
python evaluation/phase_router_vs_hash.py --skip-plots
```

Inspect a small summary directly in Python:

```python
import json
from pathlib import Path

# Load two-phase adversarial results
out = Path("test_output")
results_md = out / "two_phase_adversarial_results.md"

# Print first few lines of the Markdown table
with open(results_md) as f:
    lines = f.readlines()

print("".join(lines[:10]))  # first 10 lines of table + headers
```

**Output Example:**

```
| k | phase2_max_load | phase2_skew | hash2_max_load | hash2_skew | phase_time_ms | hash_time_ms |
|---|----------------|------------|----------------|------------|---------------|--------------|
| 16 | 18 | 1.13 | 32 | 2.50 | 12 | 8 |
| 64 | 72 | 1.12 | 128 | 2.00 | 15 | 10 |
...
**System Specs:**
- OS: Linux 5.19
- Architecture: x86_64
- Logical CPUs: 16
- Physical cores: 8
- CPU frequency: 4500.0 MHz
- Total RAM: 62.91 GB
```

> This allows you to quickly verify routing skew, max load, and system specs without opening CSV files or plotting figures.

---
