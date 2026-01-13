"""
MoE Routing Demo: Practical Capacity Planning for Mixture-of-Experts Systems

This demo shows how to use the Bit-Packed Phase Router for real-world MoE applications.
It translates from matrix operations to practical MoE concepts like experts, capacity,
and overflow risk.

The demo covers:
1. Choosing optimal k (routes per token)
2. Planning expert capacity with confidence intervals
3. Analyzing overflow risk
4. Handling real-world scenarios
"""

import numpy as np
import sys
import os

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.router_stats import (
    validate_routing_inputs,
    monte_carlo_stats,
    suggest_k_for_balance,
    estimate_expert_capacity,
    analyze_routing_distribution
)

def generate_moe_matrices(num_experts: int, tokens_per_batch: int, expert_size: int = 128) -> tuple:
    """
    Generate realistic MoE matrices.

    Args:
        num_experts: Number of experts in the MoE layer
        tokens_per_batch: Number of tokens to route
        expert_size: Size of each expert's vocabulary

    Returns:
        S, T: Binary matrices representing token-expert assignments
    """
    # For MoE, we need square matrices: tokens_per_batch x tokens_per_batch
    # where each "expert" represents a block of the matrix
    N = tokens_per_batch  # Make matrices square

    # Create S: tokens -> expert vocabulary (sparse)
    S = np.zeros((N, N), dtype=np.uint8)

    # Each token connects to a few experts' vocabulary
    experts_per_token = max(3, min(5, num_experts // 4))  # 3-5 experts
    vocab_per_expert = N // num_experts  # Vocabulary size per expert

    for token_idx in range(N):
        # Choose experts for this token
        chosen_experts = np.random.choice(num_experts, size=experts_per_token, replace=False)
        for expert_idx in chosen_experts:
            # Choose vocabulary items from this expert
            start_idx = expert_idx * vocab_per_expert
            end_idx = min(start_idx + vocab_per_expert, N)
            if start_idx < N:
                num_items = np.random.randint(5, 15)  # 5-15 items
                vocab_items = np.random.choice(range(start_idx, end_idx),
                                             size=min(num_items, end_idx - start_idx),
                                             replace=False)
                for item in vocab_items:
                    S[token_idx, item] = 1

    # Create T: expert vocabulary -> experts (block structure)
    T = np.zeros((N, N), dtype=np.uint8)

    # Each vocabulary item maps to its expert
    for expert_idx in range(num_experts):
        start_idx = expert_idx * vocab_per_expert
        end_idx = min(start_idx + vocab_per_expert, N)
        if start_idx < N:
            T[start_idx:end_idx, start_idx:end_idx] = 1  # Block diagonal

    return S, T

def moe_capacity_planner(num_experts: int, tokens_per_batch: int, desired_confidence: float = 0.99) -> dict:
    """
    End-to-end MoE capacity planning with practical recommendations.

    Args:
        num_experts: Number of experts in the MoE layer
        tokens_per_batch: Number of tokens to route per batch
        desired_confidence: Confidence level for capacity planning (e.g., 0.99)

    Returns:
        Comprehensive capacity planning report
    """
    print(f"\n MoE Capacity Planner")
    print(f"=======================")
    print(f"Experts: {num_experts}")
    print(f"Tokens per batch: {tokens_per_batch}")
    print(f"Confidence: {desired_confidence*100:.0f}%")

    # Step 1: Generate realistic MoE matrices
    print(f"\n1. Generating MoE matrices...")
    S, T = generate_moe_matrices(num_experts, tokens_per_batch)

    # Step 2: Validate inputs
    print(f"2. Validating inputs...")
    validation = validate_routing_inputs(S, T, k=16)
    if validation['warnings']:
        print(f"    Input warnings:")
        for warning in validation['warnings']:
            print(f"      - {warning}")

    # Step 3: Find optimal k (routes per token)
    print(f"3. Finding optimal k for load balance...")
    k_result = suggest_k_for_balance(S, T, target_skew=1.5, min_k=4, max_k=32,
                                    samples_per_k=8, verbose=False)
    optimal_k = k_result['recommended_k']

    print(f"   Recommended k: {optimal_k}")
    print(f"   Achieved skew: {k_result['achieved_skew']:.2f}")
    print(f"   Evaluations: {k_result['num_evaluations']}")

    # Step 4: Estimate capacity requirements
    print(f"4. Estimating expert capacity...")
    capacity = estimate_expert_capacity(S, T, optimal_k, confidence=desired_confidence)

    # Step 5: Generate practical recommendations
    mean_capacity = np.mean(capacity['mean_capacity'])
    required_capacity = np.max(capacity['required_capacity'])
    headroom = np.mean(capacity['headroom'])
    utilization = np.mean(capacity['utilization'])

    print(f"   Mean expert load: {mean_capacity:.1f} tokens")
    print(f"   Required capacity: {required_capacity:.1f} tokens")
    print(f"   Headroom needed: {headroom:.1f} tokens")
    print(f"   Utilization: {utilization*100:.1f}%")

    # Step 6: Analyze parameter sensitivity
    print(f"5. Analyzing parameter sensitivity...")
    analysis = analyze_routing_distribution(S, T, k_values=[4, 8, 16, 32], samples_per_k=5)

    print(f"   k | skew | density | mean_load | max_load")
    print(f"   --|------|---------|-----------|---------")
    for result in analysis['analysis_results']:
        print(f"   {result['k']:2d} | {result['global_skew']:.2f} | {result['edge_density']:.3f} | "
              f"{result['mean_load']:9.1f} | {result['max_load']:7.1f}")

    # Final recommendations
    recommendations = {
        'num_experts': num_experts,
        'tokens_per_batch': tokens_per_batch,
        'optimal_k': optimal_k,
        'mean_expert_load': float(mean_capacity),
        'required_capacity': float(required_capacity),
        'headroom': float(headroom),
        'utilization': float(utilization),
        'overflow_risk': 1.0 - desired_confidence,
        'parameter_analysis': analysis,
        'input_validation': validation,
        'recommendation': (
            f"Configure your MoE layer with {optimal_k} routes per token. "
            f"Allocate {int(required_capacity)} capacity per expert to handle "
            f"{desired_confidence*100:.0f}% of loads with {utilization*100:.1f}% utilization. "
            f"This provides {headroom:.1f} tokens of headroom for overflow scenarios."
        )
    }

    print(f"\n RECOMMENDATION:")
    print(f"=================")
    print(f"{recommendations['recommendation']}")

    return recommendations

def main():
    """Run the MoE routing demo with practical examples"""
    print(" MoE Routing Demo")
    print("===================")
    print("Practical capacity planning for Mixture-of-Experts systems")
    print("using the Bit-Packed Phase Router")

    # Example 1: Small MoE layer
    print(f"\n Example 1: Small MoE Layer")
    print(f"=============================")
    small_result = moe_capacity_planner(num_experts=32, tokens_per_batch=256, desired_confidence=0.95)

    # Example 2: Medium MoE layer
    print(f"\n Example 2: Medium MoE Layer")
    print(f"==============================")
    medium_result = moe_capacity_planner(num_experts=128, tokens_per_batch=1024, desired_confidence=0.99)

    # Example 3: Large MoE layer
    print(f"\n Example 3: Large MoE Layer")
    print(f"=============================")
    large_result = moe_capacity_planner(num_experts=256, tokens_per_batch=4096, desired_confidence=0.99)

    # Summary comparison
    print(f"\n Summary Comparison")
    print(f"===================")
    print(f"Config          | Optimal k | Capacity | Utilization | Overflow Risk")
    print(f"----------------|-----------|----------|-------------|---------------")
    print(f"Small (32,256)  | {small_result['optimal_k']:9d} | {small_result['required_capacity']:8.0f} | "
          f"{small_result['utilization']*100:11.1f}% | {small_result['overflow_risk']*100:13.1f}%")
    print(f"Medium (128,1K) | {medium_result['optimal_k']:9d} | {medium_result['required_capacity']:8.0f} | "
          f"{medium_result['utilization']*100:11.1f}% | {medium_result['overflow_risk']*100:13.1f}%")
    print(f"Large (256,4K)  | {large_result['optimal_k']:9d} | {large_result['required_capacity']:8.0f} | "
          f"{large_result['utilization']*100:11.1f}% | {large_result['overflow_risk']*100:13.1f}%")

    print(f"\n Demo complete!")
    print(f"   The Bit-Packed Phase Router is now ready for MoE deployment.")
    print(f"   Use the recommendations above to configure your MoE layers.")

if __name__ == "__main__":
    main()
