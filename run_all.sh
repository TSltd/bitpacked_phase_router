#!/usr/bin/env bash
set -e

# ------------------------------
# Determine maximum memory
# ------------------------------
if [ "$(uname)" = "Linux" ]; then
    # Total RAM in bytes
    HOST_MEM_BYTES=$(grep MemTotal /proc/meminfo | awk '{print $2 * 1024}')
    export MAX_MEMORY_GB=$(awk "BEGIN {printf \"%.1f\", $HOST_MEM_BYTES/1024/1024/1024}")
    echo "Detected Linux host memory: $MAX_MEMORY_GB GB"
else
    # On Windows (Docker Desktop), user must configure memory manually
    export MAX_MEMORY_GB=undefined
    echo "Windows detected. Set Docker Desktop memory manually."
fi

echo "Using MAX_MEMORY_GB=$MAX_MEMORY_GB"

# ------------------------------
# Run stress tests
# ------------------------------
python evaluation/phase_router_run.py
python evaluation/phase_router_vs_hash.py
python evaluation/phase_router_test_matrix.py
