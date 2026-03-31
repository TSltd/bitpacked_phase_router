#!/usr/bin/env bash
set -e

# Usage: ./scripts/run_full_pipeline.sh

RESULTS_DIR="results/compare_original_vs_interval"

clean() {
    rm -rf build/
    rm -f router*.so
    rm -f *.so
}

echo "========================================"
echo " PHASE 1: GROUND TRUTH (FORCED KERNELS)"
echo "========================================"

# -------------------------
# ORIGINAL
# -------------------------
clean
export CXXFLAGS="-DFORCE_ORIGINAL"
python setup.py build_ext --inplace
python evaluation/compare_original_vs_interval.py --tag original

# -------------------------
# INTERVAL
# -------------------------
clean
export CXXFLAGS="-DFORCE_INTERVAL"
python setup.py build_ext --inplace
python evaluation/compare_original_vs_interval.py --tag interval

unset CXXFLAGS

echo "========================================"
echo " KERNEL COMPARISON"
echo "========================================"

python evaluation/compare_original_vs_interval.py --compare original interval

echo "========================================"
echo " PHASE 2: DEFAULT DISPATCH (FIXED RULE)"
echo "========================================"

# -------------------------
# DEFAULT DISPATCH
# -------------------------
clean
python setup.py build_ext --inplace
python evaluation/compare_original_vs_interval.py --tag dispatch

echo "========================================"
echo " DISPATCH VS GROUND TRUTH"
echo "========================================"

python evaluation/compare_original_vs_interval.py --compare original dispatch

echo "========================================"
echo " DONE"
echo "========================================"