#!/usr/bin/env bash

set -e  # stop on error

# Usage: ./scripts/run_calibration.sh

ROOT_DIR="$(pwd)"
RESULTS_DIR="$ROOT_DIR/results"
BUILD_DIR="$ROOT_DIR/build_tmp"

PYTHON=python3

mkdir -p "$RESULTS_DIR"
mkdir -p "$BUILD_DIR"

echo "========================================"
echo " CLEAN PREVIOUS BUILDS"
echo "========================================"

rm -rf "$BUILD_DIR"/*
rm -f router*.so

# ------------------------------------------------------------
# Helper: build module with flag
# ------------------------------------------------------------
build_router () {
    MODE=$1

    echo ""
    echo "========================================"
    echo " BUILD: $MODE"
    echo "========================================"

    rm -f router*.so
    rm -rf build

    unset CXXFLAGS

    if [ "$MODE" == "interval" ]; then
        export CXXFLAGS="-DFORCE_INTERVAL -O3 -march=native -fopenmp"
    elif [ "$MODE" == "original" ]; then
        export CXXFLAGS="-DFORCE_ORIGINAL -O3 -march=native -fopenmp"
    else
        export CXXFLAGS="-DCALIBRATE_BOTH -O3 -march=native -fopenmp"
    fi
    
    $PYTHON setup.py build_ext --inplace

    # rename output to avoid import collisions
    SO_FILE=$(ls router*.so)
    cp "$SO_FILE" "$BUILD_DIR/router_${MODE}_$(basename $SO_FILE)"
}

# ------------------------------------------------------------
# Helper: run experiment in clean interpreter
# ------------------------------------------------------------
run_experiment () {
    MODE=$1

    echo ""
    echo "========================================"
    echo " RUN: $MODE"
    echo "========================================"

    # remove old modules
    rm -f "$ROOT_DIR"/router*.so

    # restore correct compiled module name
    SO_NAME=$(ls "$BUILD_DIR"/router_${MODE}_*.so)
    BASENAME=$(basename "$SO_NAME")

    # strip the router_${MODE}_ prefix
    REAL_NAME=${BASENAME#router_${MODE}_}

    cp "$SO_NAME" "$ROOT_DIR/$REAL_NAME"

    # clean python cache
    rm -rf __pycache__
    rm -rf evaluation/__pycache__

    # run fresh process
    PYTHONPATH="$ROOT_DIR" $PYTHON evaluation/density_sweep.py

    # move results
    mv results/density_sweep/density_sweep.json \
       "$RESULTS_DIR/density_sweep_${MODE}.json"

    mv results/density_sweep/equal_work.json \
       "$RESULTS_DIR/equal_work_${MODE}.json"
}

# ------------------------------------------------------------
# 1. INTERVAL CALIBRATION
# ------------------------------------------------------------
build_router interval
run_experiment interval

# ------------------------------------------------------------
# 2. ORIGINAL CALIBRATION
# ------------------------------------------------------------
build_router original
run_experiment original

# ------------------------------------------------------------
# 3. HYBRID (REAL SYSTEM)
# ------------------------------------------------------------
build_router hybrid
run_experiment hybrid

echo ""
echo "========================================"
echo " DONE"
echo "========================================"

echo "Datasets saved in:"
echo "  $RESULTS_DIR"