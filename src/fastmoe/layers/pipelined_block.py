import torch
import torch.nn as nn
from torch.profiler import record_function


class PipelinedMoEBlock(nn.Module):
    def __init__(
        self, attn_layer: nn.Module, moe_layer: nn.Module, hidden_dim: int, block_idx: int = 0
    ):
        super().__init__()
        self.attn = attn_layer
        self.moe = moe_layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.block_idx = block_idx

        self.use_streams = torch.cuda.is_available()
        if self.use_streams:
            self.moe_stream = torch.cuda.Stream()
            self.moe_in_ready = torch.cuda.Event(enable_timing=False)
            self.moe_out_done = torch.cuda.Event(enable_timing=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if not self.use_streams or B < 2:
            return self._forward_sequential(x)

        x1, x2 = x.chunk(2, dim=0)

        # -----------------------------------------------------------------
        # 1. Submit Attn(1) [Main Stream]
        # -----------------------------------------------------------------
        with record_function(f"Block {self.block_idx}: Attn(1) [Compute]"):
            residual_1 = x1
            x1_mid = residual_1 + self.attn(self.norm1(x1))
            moe_in_1 = self.norm2(x1_mid)

        self.moe_in_ready.record(torch.cuda.current_stream())

        # -----------------------------------------------------------------
        # 2. Submit Attn(2) [Main Stream]
        # -----------------------------------------------------------------
        with record_function(f"Block {self.block_idx}: Attn(2) [Compute]"):
            residual_2 = x2
            x2_mid = residual_2 + self.attn(self.norm1(x2))
            moe_in_2 = self.norm2(x2_mid)

        # -----------------------------------------------------------------
        # 3. Launch MoE(1) [Side Stream]
        # -----------------------------------------------------------------
        with torch.cuda.stream(self.moe_stream):
            self.moe_stream.wait_event(self.moe_in_ready)

            with record_function(f"Block {self.block_idx}: MoE(1) [OVERLAP]"):
                # Optional: Network Simulation
                if (
                    hasattr(self, "simulated_network_latency_ms")
                    and self.simulated_network_latency_ms > 0
                ):
                    from fastmoe.layers.pipelined_block import simulate_network_delay

                    simulate_network_delay(self.simulated_network_latency_ms, self.moe_stream)

                moe_out_1 = self.moe(moe_in_1)

            self.moe_out_done.record(self.moe_stream)
            x1_final = x1_mid + moe_out_1

        # -----------------------------------------------------------------
        # 4. Launch MoE(2) [Main Stream]
        # -----------------------------------------------------------------
        # Now MoE(2) starts immediately after Attn(2) finishes.

        with record_function(f"Block {self.block_idx}: MoE(2) [Compute]"):
            moe_out_2 = self.moe(moe_in_2)
            x2_final = x2_mid + moe_out_2

        # -----------------------------------------------------------------
        # 5. Join (Sync)
        # -----------------------------------------------------------------
        # We only wait right before we need to touch x1_final
        torch.cuda.current_stream().wait_event(self.moe_out_done)

        return torch.cat([x1_final, x2_final], dim=0)

    def _forward_sequential(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        x = resid + self.attn(self.norm1(x))
        resid = x
        x = resid + self.moe(self.norm2(x))
        return x
