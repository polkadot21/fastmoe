import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_total_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        normalize_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_total_experts = num_total_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.normalize_topk_prob = normalize_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

        self.gate = nn.Linear(hidden_dim, num_total_experts, bias=False)

    def _cap(self, num_tokens: int) -> int:
        # Minimum capacity 4 to avoid empty tensor issues
        if self.capacity_factor < 0:
            return 4  # Debug/Padding mode
        return max(
            int(math.ceil(num_tokens * self.top_k / self.num_total_experts * self.capacity_factor)),
            4,
        )

    def forward(self, x: torch.Tensor):
        """
        x: [N, D]
        """
        # 1. Routing
        logits = self.gate(x)
        scores = logits.sigmoid()  # DeepSeek V3 uses Sigmoid

        topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)

        if self.normalize_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor

        # 2. Permutation (Flattened N*K to handle collisions)
        N, K = topk_indices.shape
        capacity = self._cap(N)

        idx_flat = topk_indices.view(-1)
        w_flat = topk_weights.view(-1)

        # Priority: How many times has this expert been selected *before* this token?
        # We flatten [N, K] -> [N*K] so that the 2nd choice of Token 0 doesn't collide
        # with the 1st choice of Token 1 if they pick the same expert.
        em = F.one_hot(idx_flat, self.num_total_experts).to(torch.int32)
        pri = torch.cumsum(em, dim=0)

        active_pri = pri[torch.arange(N * K, device=x.device), idx_flat]

        # Filter by Capacity
        valid_mask = active_pri <= capacity

        # Destination: ExpertID * Cap + SlotIndex
        dest_idx = idx_flat * capacity + (active_pri - 1)
        valid_dest = dest_idx[valid_mask]

        # Gather Index: Map [Expert_Slot] -> [Original_Token_ID]
        gather_index = torch.full(
            (self.num_total_experts * capacity,), -1, dtype=torch.long, device=x.device
        )
        orig_tokens = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, K).reshape(-1)

        # Scatter the original token IDs to their destination slots
        gather_index.scatter_(0, valid_dest, orig_tokens[valid_mask])

        # 3. Create Permuted Inputs
        # We use the gather_index to pull data from x.
        # Note: We must handle the -1 indices (padding).
        safe_indices = gather_index.clone()
        safe_indices[gather_index == -1] = 0

        permuted_inputs = x[safe_indices]
        permuted_inputs[gather_index == -1] = 0.0

        # 4. Create Permuted Weights
        # We scatter the weights to the same destination
        permuted_weights = torch.zeros(
            self.num_total_experts * capacity, dtype=topk_weights.dtype, device=x.device
        )
        permuted_weights.scatter_(0, valid_dest, w_flat[valid_mask])

        return permuted_inputs, permuted_weights, gather_index, capacity
