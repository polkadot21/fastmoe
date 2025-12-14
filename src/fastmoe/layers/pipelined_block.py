import torch
import torch.nn as nn
from torch.profiler import record_function

from fastmoe.streams import MoEStreamManager


class PipelinedMoEBlock(nn.Module):
    def __init__(
        self,
        attn_layer: nn.Module,
        moe_layer: nn.Module,
        hidden_dim: int,
        manager: MoEStreamManager,
        block_idx: int = 0,
    ):
        super().__init__()
        self.attn = attn_layer
        self.moe = moe_layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.block_idx = block_idx
        self.manager = manager  # Store reference

        # Simulation Setting
        self.simulated_network_latency_ms = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # If batch is too small or no GPU, run sequential
        if not x.is_cuda or B < 2:
            return self._forward_sequential(x)

        x1, x2 = x.chunk(2, dim=0)

        # -----------------------------------------------------------------
        # 1. Attn(1) [Main Stream]
        # -----------------------------------------------------------------
        with record_function(f"Block {self.block_idx}: Attn(1) [Compute]"):
            residual_1 = x1
            x1_mid = residual_1 + self.attn(self.norm1(x1))
            moe_in_1 = self.norm2(x1_mid)

        # Signal: Input 1 is ready for Side Stream
        self.manager.record_ready()

        # -----------------------------------------------------------------
        # 2. Attn(2) [Main Stream] - Submit Early!
        # -----------------------------------------------------------------
        with record_function(f"Block {self.block_idx}: Attn(2) [Compute]"):
            residual_2 = x2
            x2_mid = residual_2 + self.attn(self.norm1(x2))
            moe_in_2 = self.norm2(x2_mid)

        # -----------------------------------------------------------------
        # 3. MoE(1) [Side Stream]
        # -----------------------------------------------------------------
        with torch.cuda.stream(self.manager.moe_stream):
            # Wait for data to be ready
            self.manager.wait_side()

            with record_function(f"Block {self.block_idx}: MoE(1) [OVERLAP]"):
                # Network Simulation Hook
                if self.simulated_network_latency_ms > 0:
                    from fastmoe.layers.pipelined_block import simulate_network_delay

                    simulate_network_delay(
                        self.simulated_network_latency_ms, self.manager.moe_stream
                    )

                moe_out_1 = self.moe(moe_in_1)

            # Signal: Output 1 is done
            self.manager.record_done()

            # This addition happens on Side Stream.
            # Note: x1_mid was computed on Main, but we waited for it. Safe.
            x1_final = x1_mid + moe_out_1

        # -----------------------------------------------------------------
        # 4. MoE(2) [Main Stream] - Unblocked!
        # -----------------------------------------------------------------
        # No waiting here. Parallelism happens now.
        with record_function(f"Block {self.block_idx}: MoE(2) [Compute]"):
            moe_out_2 = self.moe(moe_in_2)
            x2_final = x2_mid + moe_out_2

        # -----------------------------------------------------------------
        # 5. Join
        # -----------------------------------------------------------------
        # Main Stream waits for Side Stream to finish x1_final
        self.manager.wait_main()

        return torch.cat([x1_final, x2_final], dim=0)

    def _forward_sequential(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        x = resid + self.attn(self.norm1(x))
        resid = x
        x = resid + self.moe(self.norm2(x))
        return x
