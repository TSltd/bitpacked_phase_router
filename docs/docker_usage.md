## Usage

### Build Docker Image

```bash
docker build -t phase_router_tests .
```

---

### Run Full Benchmarks

```bash
docker run --rm -v "$(pwd -W 2>/dev/null || pwd)/results:/app/results" phase_router_tests
```

> This will run all three tests back-to-back and save all CSV, JSON, and Markdown results to the `results/` folder in your project directory.

**Notes:**

- **Linux/macOS:** The container will use your system’s available memory automatically.
- **Windows (Docker Desktop):** Make sure Docker has enough memory allocated:
  1. Open Docker Desktop → Settings → Resources → Memory
  2. Set memory to the maximum your system can provide (e.g., 16–32 GB).

- All plots, CSV, JSON, and Markdown summaries will be included in the `results/` folder after completion.

---

### Quick Test (Optional)

Run a **small single test** (N=256, k=32) without executing the full benchmark:

```bash
docker run --rm -v "$(pwd | tr '\\' '/' | sed 's/://g')/results:/app/results" phase_router_tests \
    python evaluation/phase_router_test.py
```

**Notes:**

- Results are saved in:

  ```
  results/phase_router_test/quick_test/
  ```

- Includes JSON metrics and PBM→PNG visualizations.
- Useful for development, CI checks, or verifying Docker setup.
- Much faster than the full benchmark.

---
