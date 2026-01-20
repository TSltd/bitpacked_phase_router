### Usage

#### 1. Build Docker Image

```bash
docker build -t phase_router_tests .
```

#### 2. Run All Benchmarks (Cross-Platform)

This runs all three tests back-to-back (`phase_router_run.py`, `phase_router_vs_hash.py`, `phase_router_test_matrix.py`) and saves all outputs to the `results/` folder in your project directory.

```bash
docker run --rm -v "$(pwd | tr '\\' '/' | sed 's/://g')/results:/app/results" phase_router_tests
```

- **Linux/macOS:** Uses your system’s available memory automatically.
- **Windows (Docker Desktop):** Make sure Docker has enough memory allocated:
  1. Open Docker Desktop → Settings → Resources → Memory
  2. Set memory to the maximum your system can provide (e.g., 16–32 GB)

#### 3. Output

All results will appear in the `results/` folder:

```
results/
├── phase_router_run/           # Scaling experiment outputs (JSON, CSV)
├── phase_router_vs_hash/       # vs. hash tests
├── phase_router_test/          # Quick single-phase test outputs
├── summary.csv                 # Combined CSV summary
├── figures/                    # Performance and scaling plots
└── reproducibility/            # Reproducibility test results
```

> All JSON, CSV, plots, and PBM→PNG conversions are handled automatically.

---

### Quick Test (Optional)

If you just want to run a **single small test** without executing the full benchmark suite, you can use `phase_router_test.py`:

```bash
docker run --rm -v "$(pwd | tr '\\' '/' | sed 's/://g')/results:/app/results" phase_router_tests \
    python evaluation/phase_router_test.py
```

**Notes:**

- This will run a **small N=256, k=32 test** and save results in:

  ```
  results/phase_router_test/quick_test/
  ```

- JSON metrics and PBM→PNG visualizations are included.
- Useful for development, CI checks, or verifying that Docker is working correctly.
- It will **not** run the full scaling experiments, so it’s much faster than the full benchmark.

---
