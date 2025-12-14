import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from fastmoe import consts
from fastmoe.distributed import Communicator, PytorchCommunicator

# We don't use the kernel here directly in the python-level logic anymore
# but we need the import if you plan to use it inside the experts loop (optional)


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        ff_dim,
        num_experts=8,
        top_k=2,
        implementation=consts.MoEImplementation.FAST,
        communicator: Communicator = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.implementation = implementation
        self.comm = communicator if communicator else PytorchCommunicator()

        # Distributed Setup
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Validate setup
        assert num_experts % self.world_size == 0, "Total experts must divide evenly by world size"
        self.num_local_experts = num_experts // self.world_size

        # Determine which experts live on this rank
        # e.g., Rank 0 gets [0, 1], Rank 1 gets [2, 3]
        self.local_expert_indices = [
            self.rank * self.num_local_experts + i for i in range(self.num_local_experts)
        ]

        # Create Experts (MLPs)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ff_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(ff_dim, dim, bias=False),
                )
                for _ in range(self.num_local_experts)
            ]
        )

        self.router = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor, group=None) -> torch.Tensor:
        """
        Full Distributed MoE Forward Pass with Detailed Histogram Exchange.

        Args:
            x: Input tensor [B, T, D]
            group: The NCCL ProcessGroup to use for communication.
                   If None, uses the default global group.
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        # ---------------------------------------------------------------------
        # 1. Routing
        # ---------------------------------------------------------------------
        logits = self.router(x_flat)
        router_probs = F.softmax(logits, dim=-1)
        # indices: [TotalTokens, K], weights: [TotalTokens, K]
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Flatten routing decisions
        indices_flat = indices.view(-1)

        # ---------------------------------------------------------------------
        # 2. Metadata: Calculate & Exchange Histogram
        # ---------------------------------------------------------------------
        # We need to know EXACTLY how many tokens go from Rank_i to Expert_j.

        # A. Calculate Global Histogram (My tokens -> All Experts)
        # [NumExperts]
        global_expert_hist = torch.bincount(indices_flat, minlength=self.num_experts).int()

        # B. Reshape to [WorldSize, LocalExperts]
        # Row i contains counts for experts residing on Rank i
        # expert_hist_per_rank[i][j] = "How many of MY tokens go to Rank i's Expert j"
        expert_hist_per_rank = global_expert_hist.view(self.world_size, self.num_local_experts)

        # C. Calculate Send Counts (Total per Rank)
        send_counts = expert_hist_per_rank.sum(dim=1).long()

        # D. Exchange Detailed Histograms
        # Note: We currently don't support custom 'group' for this metadata exchange
        # in the Communicator interface yet, but metadata is tiny so blocking is negligible.
        recv_expert_hist = self.comm.exchange_expert_histogram(expert_hist_per_rank)

        # E. Calculate Recv Counts (Total per Rank) for the heavy payload
        recv_counts = recv_expert_hist.sum(dim=1).long()

        # ---------------------------------------------------------------------
        # 3. Permute & Dispatch (Heavy Comm)
        # ---------------------------------------------------------------------
        # Sort tokens by Destination Rank, then by Destination Expert
        sorted_indices = torch.argsort(indices_flat)
        x_permuted = x_flat.repeat_interleave(self.top_k, dim=0)[sorted_indices]

        # Send data!
        # recv_data is ordered by Sending Rank: [From_Rank0 ... | From_Rank1 ...]
        # We pass the specific 'group' here to allow parallel transfers.
        recv_data = self.all_to_all(
            x_permuted, send_counts.tolist(), recv_counts.tolist(), group=group
        )

        # ---------------------------------------------------------------------
        # 4. Computation (Experts)
        # ---------------------------------------------------------------------
        # recv_data contains tokens from ALL ranks.
        # We use recv_expert_hist.view(-1) to split this flat buffer into specific chunks.
        chunks = recv_data.split(recv_expert_hist.view(-1).tolist())

        expert_outputs = []

        # For each local expert...
        for i in range(self.num_local_experts):
            # Gather all chunks destined for this expert from all ranks
            # e.g. for Exp0: we take chunk 0 (from R0), chunk 0+N (from R1), etc.
            current_expert_chunks = []
            for r in range(self.world_size):
                chunk_idx = r * self.num_local_experts + i
                current_expert_chunks.append(chunks[chunk_idx])

            # Concat -> Compute -> Split back
            expert_input = torch.cat(current_expert_chunks, dim=0)

            if expert_input.shape[0] > 0:
                out = self.experts[i](expert_input)
            else:
                out = expert_input  # Empty tensor

            # Split back into chunks to preserve order for the return trip.
            # We need to send back [Result_to_Rank0, Result_to_Rank1...]
            split_sizes = recv_expert_hist[:, i].tolist()

            if sum(split_sizes) > 0:
                split_out = out.split(split_sizes)
                expert_outputs.append(split_out)
            else:
                expert_outputs.append([torch.empty(0, self.dim, device=x.device)] * self.world_size)

        # ---------------------------------------------------------------------
        # 5. Reassemble & Combine (Return Trip)
        # ---------------------------------------------------------------------
        # We have results grouped by Expert. We need to group by Destination Rank.

        to_send_back = []
        for r in range(self.world_size):
            # For Rank 'r', collect chunks from all my local experts
            rank_chunks = []
            for i in range(self.num_local_experts):
                rank_chunks.append(expert_outputs[i][r])

            if rank_chunks:
                to_send_back.append(torch.cat(rank_chunks, dim=0))
            else:
                to_send_back.append(torch.empty(0, self.dim, device=x.device))

        final_out_buffer = torch.cat(to_send_back, dim=0)

        # Reverse All-to-All (Pass 'group' again!)
        combined_output = self.all_to_all(
            final_out_buffer, recv_counts.tolist(), send_counts.tolist(), group=group
        )

        # ---------------------------------------------------------------------
        # 6. Un-permute & Reduce
        # ---------------------------------------------------------------------
        # Restore original order [N, K, D]
        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(
            sorted_indices.size(0), device=sorted_indices.device
        )

        restored_x = combined_output[inverse_indices]

        # Weighted Sum
        restored_x = restored_x.view(B, T, self.top_k, D)

        # Reshape weights from [B*T, K] to [B, T, K, 1] so it broadcasts correctly
        weights = weights.view(B, T, self.top_k, 1)

        output = (restored_x * weights).sum(dim=2)

        return output

    def all_to_all(self, x, send_counts, recv_counts, group=None):
        """
        Helper to handle the communication using specific process groups.
        """
        if self.world_size == 1:
            return x

        # Prepare output buffer based on recv_counts
        total_recv = sum(recv_counts)
        output = torch.empty(total_recv, x.size(1), device=x.device, dtype=x.dtype)

        # We must use split/cat because dist.all_to_all_single expects list of tensors
        # for inputs/outputs if using uneven splits (which is the case here).
        # However, Pytorch's all_to_all_single handles flattened buffers with split_sizes.

        input_split_sizes = send_counts
        output_split_sizes = recv_counts

        # Pytorch's dist.all_to_all_single signature:
        dist.all_to_all_single(
            output,
            x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

        return output
