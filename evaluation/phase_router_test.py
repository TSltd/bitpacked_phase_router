"""
phase_router_test.py

Single-phase testing module for the bit-packed phase router.
Implements correctness validation, statistical analysis, and performance metrics
for the deterministic phase-separated routing algorithm.

This module uses the single-phase routing approach from router.cpp.
"""

import numpy as np
import time
from pathlib import Path
import json
from typing import Dict, Tuple, List, Optional

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import router

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available - PBM to PNG conversion disabled")


# ============================================================================
# Matrix Generation
# ============================================================================

def generate_random_binary_matrices(N: int, k_max: int, 
                                   seed_S: Optional[int] = 42, 
                                   seed_T: Optional[int] = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random binary matrices with random row counts.
    
    Args:
        N: Matrix size (N x N)
        k_max: Maximum number of ones per row
        seed_S: Random seed for S matrix (None for random)
        seed_T: Random seed for T matrix (None for random)
    
    Returns:
        S, T: Binary matrices of shape (N, N)
    """
    rng_S = np.random.default_rng(seed_S)
    rng_T = np.random.default_rng(seed_T)
    
    # Random row counts between 1 and k_max
    row_counts_S = rng_S.integers(1, k_max + 1, size=N)
    row_counts_T = rng_T.integers(1, k_max + 1, size=N)
    
    S = np.zeros((N, N), dtype=np.uint8)
    T = np.zeros((N, N), dtype=np.uint8)
    
    # Place ones randomly in each row
    for i in range(N):
        S[i, rng_S.choice(N, size=row_counts_S[i], replace=False)] = 1
        T[i, rng_T.choice(N, size=row_counts_T[i], replace=False)] = 1
    
    return S, T


# ============================================================================
# Validation Functions
# ============================================================================

def validate_routes(routes: np.ndarray, S: np.ndarray, T: np.ndarray, k: int) -> Dict[str, bool]:
    """
    Comprehensive validation of routing correctness.
    
    Checks:
    - No duplicates per row
    - All routes within bounds [0, N)
    - Routes per row <= k
    - No impossible routes (must exist in both S and T after permutation)
    
    Args:
        routes: (N, k) array of routing indices
        S: Source matrix (N, N)
        T: Target matrix (N, N)
        k: Max routes per row
    
    Returns:
        Dictionary of validation results
    """
    N = S.shape[0]
    results = {
        "valid": True,
        "no_duplicates": True,
        "within_bounds": True,
        "routes_le_k": True,
        "error_messages": []
    }
    
    for i in range(N):
        row_routes = routes[i]
        active = row_routes[row_routes >= 0]
        
        # Check for duplicates
        if len(active) != len(np.unique(active)):
            results["no_duplicates"] = False
            results["error_messages"].append(f"Row {i}: duplicate routes found")
        
        # Check bounds
        if np.any(active >= N) or np.any(active < 0):
            results["within_bounds"] = False
            results["error_messages"].append(f"Row {i}: routes out of bounds")
        
        # Check count
        if len(active) > k:
            results["routes_le_k"] = False
            results["error_messages"].append(f"Row {i}: {len(active)} routes > k={k}")
    
    results["valid"] = all([
        results["no_duplicates"],
        results["within_bounds"],
        results["routes_le_k"]
    ])
    
    return results


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_column_statistics(routes: np.ndarray, N: int) -> Dict[str, float]:
    """
    Compute column load distribution statistics.
    
    Args:
        routes: (N, k) routing array
        N: Matrix size
    
    Returns:
        Dictionary with min, max, mean, std, skew
    """
    # Build column counts
    col_counts = np.zeros(N, dtype=int)
    for i in range(routes.shape[0]):
        for j in routes[i]:
            if j >= 0:
                col_counts[j] += 1
    
    mean = np.mean(col_counts)
    std = np.std(col_counts)
    
    stats = {
        "col_min": int(np.min(col_counts)),
        "col_max": int(np.max(col_counts)),
        "col_mean": float(mean),
        "col_std": float(std),
        "col_skew": float(np.max(col_counts) / (mean + 1e-9))  # load balance ratio
    }
    
    return stats


def compute_row_statistics(routes: np.ndarray) -> Dict[str, float]:
    """
    Compute row coverage statistics.
    
    Args:
        routes: (N, k) routing array
    
    Returns:
        Dictionary with row coverage stats
    """
    row_counts = np.sum(routes >= 0, axis=1)
    
    stats = {
        "row_min": int(np.min(row_counts)),
        "row_max": int(np.max(row_counts)),
        "row_mean": float(np.mean(row_counts)),
        "row_std": float(np.std(row_counts))
    }
    
    return stats


def compute_fill_metrics(routes: np.ndarray, S: np.ndarray, T: np.ndarray) -> Dict[str, float]:
    """
    Compute fill ratio and active route counts.
    
    Args:
        routes: (N, k) routing array
        S: Source matrix
        T: Target matrix
    
    Returns:
        Dictionary with fill metrics
    """
    N, k = routes.shape
    total_capacity = N * k
    active_routes = np.sum(routes >= 0)
    
    # Total bits in source and target
    total_bits_S = np.sum(S)
    total_bits_T = np.sum(T)
    
    metrics = {
        "active_routes": int(active_routes),
        "total_capacity": int(total_capacity),
        "fill_ratio": float(active_routes / total_capacity),
        "routes_per_row": float(active_routes / N),
        "total_bits_S": int(total_bits_S),
        "total_bits_T": int(total_bits_T),
        "coverage_S": float(active_routes / total_bits_S) if total_bits_S > 0 else 0.0,
        "coverage_T": float(active_routes / total_bits_T) if total_bits_T > 0 else 0.0
    }
    
    return metrics


# ============================================================================
# Visual Utilities
# ============================================================================

def convert_pbm_to_png(pbm_folder: Path, invert: bool = True) -> List[Path]:
    """
    Convert all PBM files in folder to PNG.
    
    Args:
        pbm_folder: Path to folder containing PBM files
        invert: Whether to invert colors (white=1 becomes black)
    
    Returns:
        List of PNG file paths
    """
    if not PIL_AVAILABLE:
        return []
    
    pbm_files = sorted(pbm_folder.glob("*.pbm"))
    png_folder = pbm_folder / "png"
    png_folder.mkdir(exist_ok=True)
    
    png_files = []
    for pbm_file in pbm_files:
        try:
            im = Image.open(pbm_file).convert("L")
            if invert:
                im = Image.eval(im, lambda x: 255 - x)
            png_file = png_folder / (pbm_file.stem + ".png")
            im.save(png_file)
            png_files.append(png_file)
        except Exception as e:
            print(f"Warning: Failed to convert {pbm_file.name}: {e}")
    
    return png_files


# ============================================================================
# Main Test Function
# ============================================================================

def run_single_test(N: int, k: int, 
                   seed_S: Optional[int] = None,
                   seed_T: Optional[int] = None,
                   dump_prefix: Optional[str] = None,
                   validate: bool = True) -> Dict:
    """
    Run a single routing test and collect comprehensive metrics.
    
    Args:
        N: Matrix size
        k: Max routes per row
        seed_S: Random seed for S (None for random)
        seed_T: Random seed for T (None for random)
        dump_prefix: Folder prefix for PBM dumps (None to disable)
        validate: Whether to perform validation
    
    Returns:
        Dictionary containing all metrics and validation results
    """
    print(f"\n{'='*60}")
    print(f"Running test: N={N}, k={k}, seed_S={seed_S}, seed_T={seed_T}")
    print(f"{'='*60}")
    
    # Generate matrices
    t_gen_start = time.time()
    S, T = generate_random_binary_matrices(N, k, seed_S, seed_T)
    t_gen = (time.time() - t_gen_start) * 1000
    
    print(f"Matrix generation: {t_gen:.2f} ms")
    print(f"S bits: {np.sum(S)}, T bits: {np.sum(T)}")
    
    # Prepare routes array
    routes = np.zeros((N, k), dtype=np.int32)
    
    # Run routing with pack_and_route (includes alignment and packing)
    t_route_start = time.time()
    
    if dump_prefix:
        dump_path = Path(dump_prefix)
        dump_path.mkdir(parents=True, exist_ok=True)
        stats = router.pack_and_route(S, T, k, routes, 
                                     dump=True, 
                                     prefix=str(dump_path),
                                     validate=validate)
    else:
        stats = router.pack_and_route(S, T, k, routes, 
                                     dump=False,
                                     validate=validate)
    
    t_route = (time.time() - t_route_start) * 1000
    
    print(f"Routing completed: {t_route:.2f} ms")
    print(f"C++ reported: {stats}")
    
    # Collect all metrics
    metrics = {
        "N": N,
        "k": k,
        "seed_S": seed_S,
        "seed_T": seed_T,
        "generation_time_ms": t_gen,
        "routing_time_ms": t_route,
        "total_time_ms": t_gen + t_route
    }
    
    # Add C++ stats
    metrics.update({
        "cpp_" + key: value for key, value in stats.items()
    })
    
    # Validation
    if validate:
        validation = validate_routes(routes, S, T, k)
        metrics["validation"] = validation
        if not validation["valid"]:
            print(f"WARNING: Validation failed!")
            for msg in validation["error_messages"]:
                print(f"  - {msg}")
        else:
            print("✓ Validation passed")
    
    # Statistical analysis
    col_stats = compute_column_statistics(routes, N)
    row_stats = compute_row_statistics(routes)
    fill_metrics = compute_fill_metrics(routes, S, T)
    
    metrics.update(col_stats)
    metrics.update(row_stats)
    metrics.update(fill_metrics)
    
    print(f"\nStatistics:")
    print(f"  Active routes: {fill_metrics['active_routes']}/{N*k} ({fill_metrics['fill_ratio']*100:.1f}%)")
    print(f"  Routes per row: {fill_metrics['routes_per_row']:.2f}")
    print(f"  Column load: min={col_stats['col_min']}, max={col_stats['col_max']}, mean={col_stats['col_mean']:.2f}, skew={col_stats['col_skew']:.2f}")
    print(f"  Row coverage: min={row_stats['row_min']}, max={row_stats['row_max']}, mean={row_stats['row_mean']:.2f}")
    
    # Handle PBM -> PNG conversion
    if dump_prefix:
        dump_path = Path(dump_prefix)
        png_files = convert_pbm_to_png(dump_path, invert=True)
        metrics["png_files"] = [str(p) for p in png_files]
        print(f"  Converted {len(png_files)} PBM files to PNG")
    
    return metrics


# ============================================================================
# JSON Serialization Helper
# ============================================================================

def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ============================================================================
# Quick Test Function
# ============================================================================

if __name__ == "__main__":
    """Quick test with small N"""
    print("Running quick test with N=256, k=32")
    
    metrics = run_single_test(
        N=256,
        k=32,
        seed_S=42,
        seed_T=123,
        dump_prefix="test_output/quick_test",
        validate=True
    )
    
    # Save metrics
    output_path = Path("test_output/quick_test")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "metrics.json", "w") as f:
        json.dump(make_json_serializable(metrics), f, indent=2)
    
    print(f"\n✓ Test completed successfully!")
    print(f"Results saved to: {output_path}")
