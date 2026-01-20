"""
Transformer FFN with Learned-Seed MoE routing + iterative overflow handling.

This replaces the standard FFN:
    x -> Linear -> GELU -> Linear
with:
    x -> LearnedSeedMoE -> projection

Features:
- Deterministic, seed-based routing
- Iterative overflow handling (tries all k routes)
- Capacity-limited experts
- Drop-in compatible with Transformer blocks
- Debug and expert load reporting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import router

# ============================================================
# Seed Controller
# ============================================================

class SeedController(nn.Module):
    def __init__(self, d_model: int, num_seeds: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_seeds)
        )

    def forward(self, x: torch.Tensor) -> int:
        summary = x.mean(dim=0)
        logits = self.net(summary)
        return torch.argmax(logits).item()

# ============================================================
# Router Bank (precomputed routing patterns)
# ============================================================

class SeededRouterBank:
    def __init__(self, max_tokens: int, k: int, seeds):
        self.max_tokens = max_tokens
        self.k = k
        self.routes = {}

        S = np.ones((max_tokens, max_tokens), dtype=np.uint8)
        T = np.ones((max_tokens, max_tokens), dtype=np.uint8)

        for seed in seeds:
            routes = np.empty((max_tokens, k), dtype=np.int32)
            router.pack_and_route(
                S, T, k, routes,
                dump=False,
                validate=False,
                seed=seed
            )
            self.routes[seed] = routes

    def get(self, seed: int) -> np.ndarray:
        return self.routes[seed]

# ============================================================
# Capacity-Limited MoE FFN with iterative overflow
# ============================================================

class MoEFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        k: int,
        max_tokens: int,
        capacity_factor: float = 1.25
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.max_tokens = max_tokens

        # capacity per expert: scale by tokens * k / num_experts
        self.capacity = int(capacity_factor * np.ceil(max_tokens * k / num_experts))

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(num_experts)
        ])

        # Fallback expert for tokens that could not be routed
        self.fallback = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.last_overflow_rate = 0.0
        self.last_expert_load = None

    def forward(self, x: torch.Tensor, routes: np.ndarray, debug: bool = False) -> torch.Tensor:
        B, d = x.shape
        device = x.device

        out = torch.zeros_like(x)
        expert_load = torch.zeros(self.num_experts, device=device)
        overflow = 0

        for t in range(B):
            used = 0
            tried_experts = set()

            if debug:
                print(f"\nToken {t}: x[t][:5]={x[t, :5].tolist()}")

            for attempt in range(routes.shape[1]):
                col = routes[t, attempt]
                e = col * self.num_experts // self.max_tokens

                if e in tried_experts or e >= self.num_experts:
                    continue
                tried_experts.add(e)

                if expert_load[e] < self.capacity:
                    out[t] += self.experts[e](x[t])
                    expert_load[e] += 1
                    used += 1

                    if debug:
                        print(f"  -> Route attempt {attempt}: column={col}, expert={e}, "
                              f"expert_load={expert_load[e].item()}")

            if used == 0:
                out[t] = self.fallback(x[t])
                overflow += 1
                if debug:
                    print(f"  !! Token {t} overflowed. Using fallback.")
            else:
                out[t] /= used
                if debug:
                    print(f"  -> Token {t} routed to {used} expert(s).")

        self.last_overflow_rate = overflow / B
        self.last_expert_load = expert_load.clone()

        if debug:
            print(f"\n=== Summary ===")
            print(f"Total overflow tokens: {overflow}/{B}")
            print(f"Expert loads (first 10): {expert_load[:10].tolist()}")
            print(f"Overflow rate: {self.last_overflow_rate:.4f}")
            top_experts = torch.topk(expert_load, 10)
            print("Top 10 busiest experts:", top_experts.indices.tolist())
            print("Loads of top experts:", top_experts.values.tolist())

        return out

# ============================================================
# Transformer MoE FFN block
# ============================================================

class TransformerMoEFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        k: int,
        max_tokens: int,
        num_seeds: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seeds = list(range(num_seeds))

        self.router_bank = SeededRouterBank(
            max_tokens=max_tokens,
            k=k,
            seeds=self.seeds
        )

        self.seed_controller = SeedController(d_model=d_model, num_seeds=num_seeds)

        self.moe_ffn = MoEFFN(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            k=k,
            max_tokens=max_tokens
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        seed_idx = self.seed_controller(x)
        seed = self.seeds[seed_idx]
        routes = self.router_bank.get(seed)

        x = self.moe_ffn(x, routes, debug=debug)
        x = self.dropout(x)
        return x + residual

# ============================================================
# Demo
# ============================================================

def demo():
    torch.manual_seed(0)

    B = 256        # number of tokens
    d_model = 512
    d_ff = 2048
    num_experts = B  # match router rows
    k = 4         # routes per token
    num_seeds = 8

    # Create Transformer MoE FFN block
    ffn = TransformerMoEFFN(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        k=k,
        max_tokens=B,
        num_seeds=num_seeds
    )

    # Create dummy input
    x = torch.randn(B, d_model)

    # Run the block with debug enabled
    print("\n=== Running TransformerMoEFFN with debug ===")
    y = ffn(x, debug=True)

    print("\n=== Demo Summary ===")
    print("Output shape:", y.shape)
    print("Overflow rate:", ffn.moe_ffn.last_overflow_rate)
    print("First 10 expert loads:", ffn.moe_ffn.last_expert_load[:10].tolist())
    top_experts = torch.topk(ffn.moe_ffn.last_expert_load, 10)
    print("Top 10 busiest experts:", top_experts.indices.tolist())
    print("Loads of top experts:", top_experts.values.tolist())

if __name__ == "__main__":
    demo()
