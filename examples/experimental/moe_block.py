import torch
import torch.nn as nn
import numpy as np
import router


class CapacityMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        k: int,
        max_tokens: int,
        capacity_factor: float = 1.25,
        seed: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.max_tokens = max_tokens
        self.seed = seed

        # Capacity per expert
        self.capacity = int(
            np.ceil(capacity_factor * max_tokens / num_experts)
        )

        # Experts (identical at init)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_experts)
        ])

        # Dense fallback (for overflow)
        self.fallback = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Router matrices (square!)
        N = max_tokens
        self.S = np.ones((N, N), dtype=np.uint8)
        self.T = np.ones((N, N), dtype=np.uint8)
        self.routes = np.empty((N, k), dtype=np.int32)

        # Precompute routing
        router.pack_and_route(self.S, self.T, k, self.routes, seed=seed)

    def forward(self, x):
        """
        x: [B, d_model]
        """
        B = x.size(0)
        assert B <= self.max_tokens

        device = x.device
        out = torch.zeros_like(x)

        # Track per-expert load
        expert_load = torch.zeros(
            self.num_experts, dtype=torch.int32, device=device
        )

        overflow_count = 0

        for token_idx in range(B):
            expert_ids = self.routes[token_idx]
            expert_ids = expert_ids[expert_ids != -1]
            expert_ids = expert_ids[expert_ids < self.num_experts]

            used = 0

            for expert_id in expert_ids:
                if expert_load[expert_id] < self.capacity:
                    out[token_idx] += self.experts[expert_id](x[token_idx])
                    expert_load[expert_id] += 1
                    used += 1

            if used == 0:
                # Overflow fallback
                out[token_idx] = self.fallback(x[token_idx])
                overflow_count += 1
            else:
                out[token_idx] /= used

        # Optional: attach stats (for logging)
        self.last_overflow_rate = overflow_count / B
        self.last_expert_load = expert_load.detach().cpu()

        return out
