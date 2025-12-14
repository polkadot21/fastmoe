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
        # Fallback if batch is too small to split
        if not self.use_streams or B < 2:
            return self._forward_sequential(x)

        x1, x2 = x.chunk(2, dim=0)

        # --- STAGE 1: Main Stream (Attn Chunk 1) ---
        with record_function(f"Block {self.block_idx}: Attn(1) [Compute]"):
            residual_1 = x1
            x1_mid = residual_1 + self.attn(self.norm1(x1))
            moe_in_1 = self.norm2(x1_mid)

        # Signal ready
        self.moe_in_ready.record(torch.cuda.current_stream())

        # --- STAGE 2: Overlap ---

        # A. SIDE STREAM (The Red Bar: Comm + Expert)
        with torch.cuda.stream(self.moe_stream):
            self.moe_stream.wait_event(self.moe_in_ready)

            with record_function(f"Block {self.block_idx}: MoE(1) [OVERLAP]"):
                moe_out_1 = self.moe(moe_in_1)

            self.moe_out_done.record(self.moe_stream)
            x1_final = x1_mid + moe_out_1

        # B. MAIN STREAM (The Blue Bar: Compute Attn 2)
        with record_function(f"Block {self.block_idx}: Attn(2) [Compute]"):
            residual_2 = x2
            x2_mid = residual_2 + self.attn(self.norm1(x2))
            moe_in_2 = self.norm2(x2_mid)

        # --- STAGE 3: Join ---
        torch.cuda.current_stream().wait_event(self.moe_out_done)

        with record_function(f"Block {self.block_idx}: MoE(2) [Compute]"):
            moe_out_2 = self.moe(moe_in_2)
            x2_final = x2_mid + moe_out_2

        return torch.cat([x1_final, x2_final], dim=0)

    def _forward_sequential(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        x = resid + self.attn(self.norm1(x))
        resid = x
        x = resid + self.moe(self.norm2(x))
        return x
