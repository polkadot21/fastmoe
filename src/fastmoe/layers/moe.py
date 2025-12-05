import torch
import torch.nn as nn
import torch.nn.functional as F

from fastmoe.kernels.ops import weighted_scatter_add


class MoEFeedForward(nn.Module):
    def __init__(self, dim, ff_dim, num_experts=8, top_k=2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. Gating Network
        self.router = nn.Linear(dim, num_experts, bias=False)

        # 2. Experts
        # In a real distributed setting, these are sharded.
        # Here, we keep them local to demonstrate the scatter-add speedup.
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ff_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(ff_dim, dim, bias=False),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        # --- Routing ---
        logits = self.router(x_flat)
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).type_as(x)

        # --- Pre-Dispatch Calculation ---
        indices_flat = topk_indices.view(-1)
        weights_flat = topk_weights.view(-1)

        # "src_indices" maps the N*K tokens back to their original N row in x_flat
        src_indices = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)

        # Sort to group tokens by expert assignment
        sort_indices = torch.argsort(indices_flat)
        expert_indices_sorted = indices_flat[sort_indices]

        # We must permute the metadata to match the sorted tokens
        reverse_map_indices = src_indices[sort_indices]
        sorted_weights = weights_flat[sort_indices]

        # Dispatch: Permute data to be contiguous for each expert
        permuted_data = x_flat[reverse_map_indices]

        # --- Expert Computation ---
        expert_counts = torch.bincount(expert_indices_sorted, minlength=self.num_experts)
        tokens_per_expert = torch.split(permuted_data, expert_counts.tolist())

        results = []
        for i, expert in enumerate(self.experts):
            if tokens_per_expert[i].shape[0] > 0:
                results.append(expert(tokens_per_expert[i]))
            else:
                results.append(torch.empty(0, D, device=x.device, dtype=x.dtype))

        # This intermediate concat is unavoidable locally but smaller than the full recombine
        expert_output_flat = torch.cat(results)

        # --- Optimized Recombination ---
        # The key innovation: "No-Cat" Recombination
        # Instead of sorting `expert_output_flat` and using `index_select` or `cat`,
        # we scatter-add directly to the final buffer.
        out = weighted_scatter_add(
            expert_output_flat, reverse_map_indices, sorted_weights, (B * T, D)
        )

        return out.view(B, T, D)
