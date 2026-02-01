import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k, capacity_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Capacity = (Tokens / Experts) * CapacityFactor
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [Batch, Seq, Dim]
        B, S, D = x.shape
        num_tokens = B * S

        # 1. Gating
        x_flat = x.view(-1, D)
        logits = self.gate(x_flat)  # [Tokens, Experts]
        scores = F.softmax(logits, dim=1)

        # 2. Top-K Selection
        # topk_weights: [Tokens, K], topk_indices: [Tokens, K]
        topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=1)

        # 3. Calculate Capacity
        # How many tokens *can* each expert handle?
        # Capacity per expert
        capacity = int((num_tokens / self.num_experts) * self.capacity_factor)
        # Ensure minimal capacity
        capacity = max(capacity, 4)

        # 4. Create Indexing Masks (The Hard Part)
        # We need to find exactly 'capacity' tokens for each expert.

        # [Tokens, K] -> One-hot mask [Tokens, K, Experts]
        # We use a trick: Scatter indices to count.
        # Ideally, we want to assign each token to a slot in the expert's buffer.

        # Simple Greedy Assignment via Sort:
        # We flatten the top-k selections to treat them as independent requests
        # flat_indices: [Tokens * K] containing expert IDs
        flat_indices = topk_indices.view(-1)

        # Sort by expert ID to group them
        # sorted_expert_ids: [0, 0, 0, ..., 1, 1, ...]
        # sort_idx: The original index of the request
        sorted_expert_ids, sort_idx = torch.sort(flat_indices)

        # Now we need to figure out which requests fall within capacity.
        # We calculate the cumulative count of each expert ID.
        # This is expensive in pure Python, so we use a simplified Histogram approach for static shapes. # noqa

        # --- Optimized Static Routing ---
        # We will create a routing map: inputs[i] -> (Expert_E, Slot_S)

        # 1. Mask for each expert
        # expert_mask: [Tokens, K, Experts]
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).to(torch.int32)

        # 2. Position in Expert Buffer (CumSum)
        # token_priority: [Tokens, K, Experts] - The N-th token destined for Expert E
        # We cumsum along the Token dimension
        token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask

        # 3. Filter by Capacity
        # valid_mask: [Tokens, K, Experts] - 1 if token fits in capacity, 0 otherwise
        valid_mask = (token_priority > 0) & (token_priority <= capacity)

        # 4. Calculate Final Gather Indices
        # We want to map: (Expert_E, Slot_S) -> Input_Token_Index
        # Since we need to produce a tensor of shape [Experts, Capacity, Dim],
        # we need an index tensor of shape [Experts * Capacity] that points to original tokens.

        # Flatten masks to [Tokens * K, Experts]
        valid_mask_flat = valid_mask.view(-1, self.num_experts)
        token_priority_flat = token_priority.view(-1, self.num_experts)

        # Identify selected tokens
        # We need to construct a gather_index of shape [Num_Experts * Capacity]
        # Initialize with -1 (padding)
        gather_index = torch.full(
            (self.num_experts * capacity,), -1, dtype=torch.long, device=x.device
        )

        # We need to place 'original_token_idx' at 'expert_offset + slot_index'
        # original_token_idx: floor(row_idx / K)
        # row_idx range: 0 to Tokens*K
        row_idx = (
            torch.arange(num_tokens * self.top_k, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.num_experts)
        )
        original_token_idx = row_idx // self.top_k

        # Calculate destination index in the flat expert buffer:
        # dest_idx = Expert_ID * Capacity + (Priority - 1)
        expert_ids_range = torch.arange(self.num_experts, device=x.device).unsqueeze(0)
        dest_idx = expert_ids_range * capacity + (token_priority_flat - 1)

        # Only apply valid ones
        active = valid_mask_flat.bool()

        # Scatter the original token indices into the gather map
        # gather_index[dest_idx[active]] = original_token_idx[active]
        gather_index.scatter_(0, dest_idx[active].flatten(), original_token_idx[active].flatten())

        # Also save weights for the combine step
        # We need to store weights in the same layout as the permuted tokens
        # weights_flat: [Tokens * K]
        weights_flat = topk_weights.view(-1).unsqueeze(1).expand(-1, self.num_experts)  # Broadcast

        # permuted_weights: [Num_Experts * Capacity]
        permuted_weights = torch.zeros(
            (self.num_experts * capacity,), dtype=x.dtype, device=x.device
        )
        permuted_weights.scatter_(0, dest_idx[active].flatten(), weights_flat[active].flatten())

        # 5. Permute Inputs (Dispatch Preparation)
        # We use the gather_index to pull tokens.
        # Handle padding (-1 indices): We clamp to 0 temporarily, gather, then zero out.
        safe_gather_index = gather_index.clamp(min=0)

        # x_flat: [Tokens, Dim] -> [Experts * Capacity, Dim]
        permuted_inputs = x_flat[safe_gather_index]

        # Apply padding mask (where index was -1)
        padding_mask = (gather_index == -1).unsqueeze(1)
        permuted_inputs.masked_fill_(padding_mask, 0.0)

        return permuted_inputs, permuted_weights, gather_index, capacity
