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
        gate_type: str = "mlp",
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

        # Simple Linear Gate (standard for MoE)
        self.gate = nn.Linear(hidden_dim, num_total_experts, bias=False)

    def _cap(self, num_tokens: int) -> int:
        """Calculate capacity per expert."""
        # Minimum capacity 4 to avoid CUDAGraph/kernel issues with empty tensors
        return max(
            int(math.ceil(num_tokens * self.top_k / self.num_total_experts * self.capacity_factor)),
            4,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [Batch*Seq, Dim] - Flattened hidden states
        Returns:
            permuted_inputs: [Experts * Capacity, Dim]
            permuted_weights: [Experts * Capacity]
            gather_index: [Experts * Capacity]
            capacity: int
        """
        # 1. Routing (Standard TopK)
        logits = self.gate(x)  # [N, Experts]
        scores = logits.sigmoid()  # or softmax, depending on config. DeepSeek uses sigmoid.

        topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)

        if self.normalize_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor

        # 2. Permutation Logic (The Critical Fix)
        N, K = topk_indices.shape
        capacity = self._cap(N)

        # Flatten to [N*K] to ensure unique slot assignments across the 'K' dimension
        idx_flat = topk_indices.view(-1)
        w_flat = topk_weights.view(-1)

        # Priority / Slot Calculation
        # One-hot + Cumsum allows us to find the i-th occurrence of Expert E
        em = F.one_hot(idx_flat, self.num_total_experts).to(torch.int32)
        pri = torch.cumsum(em, dim=0)

        # Select priority for the specific chosen expert
        active_pri = pri[torch.arange(N * K, device=x.device), idx_flat]

        # Filter by capacity
        valid_mask = active_pri <= capacity

        # Calculate Destination Address: Dest = ExpertID * Capacity + (Slot - 1)
        dest_idx = idx_flat * capacity + (active_pri - 1)
        valid_dest = dest_idx[valid_mask]

        # Create Gather Index (Map: Slot -> OriginalTokenID)
        # Initialize with -1 (Padding)
        gather_index = torch.full(
            (self.num_total_experts * capacity,), -1, dtype=torch.long, device=x.device
        )

        # Create original token IDs: [0, 0, 1, 1, 2, 2 ...] corresponding to K expansion
        orig_tokens = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, K).reshape(-1)
        valid_orig = orig_tokens[valid_mask]

        # Scatter Mapping
        gather_index.scatter_(0, valid_dest, valid_orig)

        # 3. Create Permuted Inputs (Data Movement)
        # Handle padding cleanly
        safe_indices = gather_index.clone()
        safe_indices[gather_index == -1] = 0
        permuted_inputs = x[safe_indices]
        permuted_inputs[gather_index == -1] = 0.0

        # 4. Create Permuted Weights
        permuted_weights = torch.zeros(
            self.num_total_experts * capacity, dtype=topk_weights.dtype, device=x.device
        )
        valid_w = w_flat[valid_mask]
        permuted_weights.scatter_(0, valid_dest, valid_w)

        return permuted_inputs, permuted_weights, gather_index, capacity
