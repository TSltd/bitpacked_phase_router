### Usage

#### Build Docker Image

```bash
docker build -t phase_router_tests .
```

#### Run Benchmarks (cross-platform)

```bash
docker run --rm -v "$(pwd -W 2>/dev/null || pwd)/results:/app/results" phase_router_tests
```

> This will run all three tests back-to-back and save all CSV, JSON, and Markdown results to the `results/` folder in your project directory.

#### Notes

- **Linux/macOS:** The command uses your system’s available memory automatically.
- **Windows (Docker Desktop):** Ensure Docker has enough memory allocated:
  1. Open Docker Desktop → Settings → Resources → Memory
  2. Set memory to the maximum your system can provide (e.g., 16–32 GB).

All plots and summaries will be included in the `results/` folder after completion.

---
