#!/usr/bin/env bash
set -e

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
echo " PHASE 2: FIT WEIGHTS"
echo "========================================"

python evaluation/fit_dispatch_weights.py

echo "========================================"
echo " PHASE 3: EVALUATE DISPATCH"
echo "========================================"

# -------------------------
# SIMPLE RULE
# -------------------------
clean
export CXXFLAGS="-DUSE_LEARNED_MODEL=0"
python setup.py build_ext --inplace
python evaluation/compare_original_vs_interval.py --tag simple_rule

# -------------------------
# LEARNED MODEL
# -------------------------
clean
export CXXFLAGS="-DUSE_LEARNED_MODEL=1"
python setup.py build_ext --inplace
python evaluation/compare_original_vs_interval.py --tag learned_model

unset CXXFLAGS

echo "========================================"
echo " DISPATCH COMPARISON"
echo "========================================"

python evaluation/compare_original_vs_interval.py --compare simple_rule learned_model

echo "========================================"
echo " DONE"
echo "========================================"