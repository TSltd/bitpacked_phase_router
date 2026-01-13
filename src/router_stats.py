"""
router_stats.py - Monte-Carlo Statistical Utilities for Bit-Packed Phase Router

This module provides statistical analysis tools for the stochastic routing algorithm.
The router implements a Chung–Lu-style bipartite sampler that creates balanced
sparse subgraphs. These utilities enable:

- Monte-Carlo sampling of routing distributions
- Load balancing analysis
- Capacity planning for MoE systems
- Parameter optimization

The core insight: we treat the router as a Monte-Carlo transport operator that
samples random feasible couplings given degree marginals and sparsity constraints.
"""

import numpy as np
import warnings
from typing import Tuple, Dict, Optional, Union

# Import the router module
try:
    import router
    import router_py
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    warnings.warn("Router module not available - statistical functions will use mock data")

def sample_many(
    S: np.ndarray,
    T: np.ndarray,
    k: int,
    num_samples: int = 16,
    seed_base: int = 42
) -> np.ndarray:
    """
    Run multiple independent routing samples and return column loads.

    This is the fundamental Monte-Carlo primitive. Given binary matrices S and T,
    it draws `num_samples` independent samples from the Chung–Lu distribution and
    returns the column load for each sample.

    Args:
        S: Binary source matrix of shape (N, N)
        T: Binary target matrix of shape (N, N)
        k: Maximum routes per row (sparsity constraint)
        num_samples: Number of independent samples to draw
        seed_base: Base seed for reproducibility (samples use seed_base + i)

    Returns:
        column_loads: Array of shape (num_samples, N) where
                     column_loads[sample, j] = total load on column j
                     in sample `sample`

    Note:
        This returns column loads, not routes. Routes are transient sampling
        artifacts; column loads are the meaningful statistical quantities.
    """
    if not ROUTER_AVAILABLE:
        # Chung–Lu-consistent fallback sampler
        # E[L_j] = total_routes * T_j / sum(T)
        N = S.shape[0]
        tj = T.sum(axis=0).astype(np.float64)
        total_routes = k * N  # expected upper bound; used only for fallback
        lam = total_routes * tj / (tj.sum() + 1e-12)
        return np.random.poisson(lam, size=(num_samples, N))


    N = S.shape[0]
    if S.shape != T.shape:
        raise ValueError(f"S and T must have same shape, got {S.shape} vs {T.shape}")
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"Matrices must be square, got shape {S.shape}")

    column_loads = np.zeros((num_samples, N), dtype=np.int32)

    for sample in range(num_samples):
        # Create routes array for this sample
        routes = np.zeros((N, k), dtype=np.int32)

        # Run router with deterministic seed (returns void, not stats)
        router.pack_and_route(S, T, k, routes, seed=seed_base + sample)

        # Compute column loads: count how many routes point to each column
        # Vectorized implementation using bincount (10-50x faster than nested loops)
        flat_routes = routes.reshape(-1)
        valid_routes = flat_routes[flat_routes >= 0]
        column_loads[sample] = np.bincount(valid_routes, minlength=N)

    return column_loads

def monte_carlo_stats(
    S: np.ndarray,
    T: np.ndarray,
    k: int,
    num_samples: int = 32,
    seed_base: int = 42
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute comprehensive Monte-Carlo statistics for routing behavior.

    This function answers the question: "What distribution of column loads should
    I expect when using this routing configuration?"

    Args:
        S: Binary source matrix of shape (N, N)
        T: Binary target matrix of shape (N, N)
        k: Maximum routes per row
        num_samples: Number of samples for Monte-Carlo estimation
        seed_base: Base seed for reproducibility

    Returns:
        Dictionary containing:
        - mean_load: Expected column load (shape: N)
        - std_load: Standard deviation of column loads (shape: N)
        - p95_load: 95th percentile load (shape: N)
        - max_load: Maximum observed load across samples (shape: N)
        - skew: Load imbalance ratio (max/mean) per column (shape: N)
        - global_skew: Overall load imbalance (max mean / mean mean)
        - raw_samples: Raw column loads for all samples (shape: num_samples, N)

    Example:
        >>> stats = monte_carlo_stats(S, T, k=32)
        >>> print(f"Expected max load: {np.max(stats['mean_load']):.1f}")
        >>> print(f"95th percentile: {np.max(stats['p95_load']):.1f}")
        >>> print(f"Overall skew: {stats['global_skew']:.2f}")
    """
    loads = sample_many(S, T, k, num_samples, seed_base)

    # Compute basic statistics
    mean_load = np.mean(loads, axis=0)
    std_load = np.std(loads, axis=0)
    p95_load = np.percentile(loads, 95, axis=0)
    max_load = np.max(loads, axis=0)

    # Compute temporal skew (per-column load variability across samples)
    temporal_skew = np.zeros_like(mean_load, dtype=float)
    valid_mask = mean_load > 0
    temporal_skew[valid_mask] = max_load[valid_mask] / mean_load[valid_mask]
    temporal_skew[~valid_mask] = 1.0  # No load = perfect balance

    # Global skew metric (MoE overload risk: max mean load / mean of mean loads)
    global_skew = np.max(mean_load) / (np.mean(mean_load) + 1e-9)

    # Theoretical baseline: Chung–Lu expectation
    total_t = np.sum(T)
    if total_t > 0:
        # Chung–Lu expectation using observed total routing mass
        total_routes = np.sum(mean_load)
        ideal_mean = total_routes * T.sum(axis=0) / (total_t + 1e-12)
        
        bias = mean_load - ideal_mean
        relative_error = np.zeros_like(mean_load)
        valid_bias_mask = ideal_mean > 0
        relative_error[valid_bias_mask] = bias[valid_bias_mask] / ideal_mean[valid_bias_mask]
    else:
        ideal_mean = bias = relative_error = None

    return {
        'mean_load': mean_load,
        'std_load': std_load,
        'p95_load': p95_load,
        'max_load': max_load,
        'temporal_skew': temporal_skew,  # Renamed for clarity
        'global_skew': float(global_skew),
        'ideal_mean': ideal_mean,  # Theoretical Chung–Lu expectation
        'bias': bias,  # Deviation from ideal
        'relative_error': relative_error,  # Relative deviation
        'raw_samples': loads,
        'num_samples': num_samples,
        'k': k,
        'N': S.shape[0]
    }

def suggest_k_for_balance(
    S: np.ndarray,
    T: np.ndarray,
    target_skew: float = 1.5,
    min_k: int = 8,
    max_k: int = 256,
    samples_per_k: int = 10,
    seed_base: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Find the smallest k that achieves desired load balance.

    This function performs a binary search to find the minimal k such that
    the column load skew (max/mean) is ≤ target_skew. This is essentially
    solving for the sparsity budget needed to achieve a given load balancing
    constraint.

    Since global_skew(k) is monotone decreasing in k, binary search provides
    O(log(max_k - min_k)) efficiency instead of O(max_k - min_k) for linear search.

    Args:
        S, T: Binary matrices defining the routing problem
        target_skew: Desired maximum load imbalance ratio
        min_k: Minimum k to consider
        max_k: Maximum k to consider
        samples_per_k: Number of samples for estimating skew at each k
        seed_base: Base seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
        - recommended_k: Smallest k achieving target skew
        - achieved_skew: Actual skew at recommended_k
        - search_results: Full results for all tested k values
        - stats: Full statistics at recommended_k
        - num_evaluations: Number of k values tested

    Note:
        Uses binary search for efficiency (10-30x faster than linear search
        for typical parameter ranges).
    """
    N = S.shape[0]
    search_results = []
    num_evaluations = 0

    # Binary search for minimal k achieving target skew
    low, high = min_k, max_k
    best_k = None
    best_skew = float('inf')

    while low <= high:
        mid = (low + high) // 2
        num_evaluations += 1

        if verbose:
            print(f"Testing k={mid}... ", end="", flush=True)

        # Get statistics for this k
        stats = monte_carlo_stats(S, T, mid, samples_per_k, seed_base)
        current_skew = stats['global_skew']

        search_results.append({
            'k': mid,
            'global_skew': current_skew,
            'mean_load': np.mean(stats['mean_load']),
            'max_load': np.max(stats['mean_load'])
        })

        if verbose:
            print(f"skew={current_skew:.2f} (mean={np.mean(stats['mean_load']):.1f}, "
                  f"max={np.max(stats['mean_load']):.1f})")

        if current_skew <= target_skew:
            # This k works, try smaller k
            best_k = mid
            best_skew = current_skew
            high = mid - 1
            if verbose:
                print(f"  ✓ Found suitable k={mid}, searching for smaller...")
        else:
            # This k doesn't work, try larger k
            low = mid + 1

    if best_k is None:
        if verbose:
            print(f"\nWarning: No k in range [{min_k}, {max_k}] achieved target_skew={target_skew}")
        # Return the best available
        if search_results:
            best_result = min(search_results, key=lambda x: x['global_skew'])
            best_k = best_result['k']
            best_skew = best_result['global_skew']
        else:
            # No results at all, return midpoint
            best_k = (min_k + max_k) // 2
            best_skew = float('inf')

    # Get full statistics for the recommended k
    final_stats = monte_carlo_stats(S, T, best_k, samples_per_k * 2, seed_base)

    return {
        'recommended_k': best_k,
        'achieved_skew': best_skew,
        'target_skew': target_skew,
        'search_results': search_results,
        'stats': final_stats,
        'success': best_skew <= target_skew,
        'num_evaluations': num_evaluations
    }

def estimate_expert_capacity(
    S: np.ndarray,
    T: np.ndarray,
    k: int,
    confidence: float = 0.95,
    num_samples: int = 50,
    seed_base: int = 42
) -> Dict:
    """
    Estimate expert capacity requirements for MoE deployment.

    Answers the question: "How much capacity do I need to allocate to handle
    routing loads with (1-confidence) probability?"

    Args:
        S, T: Binary matrices
        k: Routes per row
        confidence: Confidence level (e.g., 0.95 for 95th percentile)
        num_samples: Number of Monte-Carlo samples
        seed_base: Base seed

    Returns:
        Dictionary containing:
        - mean_capacity: Expected load per expert
        - required_capacity: Capacity needed at specified confidence level
        - headroom: Additional capacity needed (required - mean)
        - utilization: Mean / required capacity
        - raw_samples: All samples for custom analysis

    Example:
        >>> capacity = estimate_expert_capacity(S, T, k=32, confidence=0.99)
        >>> print(f"Need {np.max(capacity['required_capacity'])} capacity "
        >>>       f"to handle 99% of loads")
    """
    loads = sample_many(S, T, k, num_samples, seed_base)

    mean_capacity = np.mean(loads, axis=0)
    required_capacity = np.percentile(loads, 100 * confidence, axis=0)
    headroom = required_capacity - mean_capacity
    utilization = mean_capacity / (required_capacity + 1e-9)

    return {
        'mean_capacity': mean_capacity,
        'required_capacity': required_capacity,
        'headroom': headroom,
        'utilization': utilization,
        'confidence_level': confidence,
        'raw_samples': loads,
        'k': k,
        'N': S.shape[0]
    }

def analyze_routing_distribution(
    S: np.ndarray,
    T: np.ndarray,
    k_values: Optional[Union[list, np.ndarray]] = None,
    samples_per_k: int = 20,
    seed_base: int = 42
) -> Dict:
    """
    Comprehensive analysis of routing behavior across multiple k values.

    This function provides a complete characterization of how routing behavior
    changes with sparsity budget k. It's useful for understanding the trade-off
    between sparsity and load balancing.

    Args:
        S, T: Binary matrices
        k_values: List of k values to analyze (default: geometric sequence)
        samples_per_k: Samples per k value
        seed_base: Base seed

    Returns:
        Comprehensive analysis with statistics for each k
    """
    N = S.shape[0]

    if k_values is None:
        # Default: geometric sequence from 4 to min(256, N)
        max_k = min(256, N)
        k_values = [int(4 * (1.5 ** i)) for i in range(20) if int(4 * (1.5 ** i)) <= max_k]

    analysis_results = []

    for k in k_values:
        stats = monte_carlo_stats(S, T, k, samples_per_k, seed_base)

        analysis_results.append({
            'k': k,
            'global_skew': stats['global_skew'],
            'mean_load': np.mean(stats['mean_load']),
            'max_load': np.max(stats['mean_load']),
            'std_load': np.mean(stats['std_load']),
            'p95_load': np.max(stats['p95_load']),
            'total_routes': np.sum(stats['mean_load']),
            'edge_density': np.sum(stats['mean_load']) / (N * N),  # Renamed for clarity
            'theoretical_density': k / N  # Expected density from theory
        })

    return {
        'analysis_results': analysis_results,
        'k_values': k_values,
        'S_shape': S.shape,
        'T_shape': T.shape,
        'samples_per_k': samples_per_k
    }

# Example usage and demonstration
if __name__ == "__main__":
    print("router_stats.py - Monte-Carlo Statistical Utilities")
    print("=" * 60)

    # Create small test matrices
    N = 128
    k_test = 16

    # Generate random binary matrices
    np.random.seed(42)
    S_test = np.random.randint(0, 2, size=(N, N))
    T_test = np.random.randint(0, 2, size=(N, N))

    print(f"Testing with N={N}, random binary matrices")
    print(f"S density: {np.mean(S_test):.3f}, T density: {np.mean(T_test):.3f}")
    print()

    # Example 1: Basic Monte-Carlo stats
    print("1. Monte-Carlo Statistics:")
    stats = monte_carlo_stats(S_test, T_test, k_test, num_samples=10)
    print(f"   Mean load: {np.mean(stats['mean_load']):.2f} ± {np.mean(stats['std_load']):.2f}")
    print(f"   Max load: {np.max(stats['mean_load']):.2f} (95th: {np.max(stats['p95_load']):.2f})")
    print(f"   Global skew: {stats['global_skew']:.2f}")
    print()

    # Example 2: Find optimal k
    print("2. Optimal k for load balance:")
    k_search = suggest_k_for_balance(S_test, T_test, target_skew=1.8,
                                    min_k=8, max_k=32, samples_per_k=5)
    print(f"   Recommended k: {k_search['recommended_k']}")
    print(f"   Achieved skew: {k_search['achieved_skew']:.2f}")
    print(f"   Success: {'Yes' if k_search['success'] else 'No'}")
    print()

    # Example 3: Capacity planning
    print("3. Expert Capacity Estimation:")
    capacity = estimate_expert_capacity(S_test, T_test, k_test, confidence=0.95)
    print(f"   Mean capacity needed: {np.mean(capacity['mean_capacity']):.2f}")
    print(f"   95th percentile: {np.mean(capacity['required_capacity']):.2f}")
    print(f"   Headroom: {np.mean(capacity['headroom']):.2f} ({100*np.mean(capacity['utilization']):.1f}% utilization)")
    print()

    print("✓ router_stats.py ready for use!")
    print("  Import with: from router_stats import monte_carlo_stats, suggest_k_for_balance, estimate_expert_capacity")
