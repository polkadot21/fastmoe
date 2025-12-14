import torch
import torch.nn as nn
from torch.cuda import nvtx


class PipelinedMoEBlock(nn.Module):
    """
    Production-Grade Pipelined MoE Block (Volodya's Approach).

    Logic:
    1. Split Batch -> [B1, B2]
    2. Main Stream: Compute Attn(B1) -> produces 'moe_in_1'.
    3. Signal Event: "moe_in_ready"
    4. Side Stream: Wait for "moe_in_ready", run MoE(B1), Signal "moe_out_done".
    5. Main Stream: Immediately run Attn(B2) (Overlap!).
    6. Main Stream: Wait for "moe_out_done", then Merge.
    """

    def __init__(self, attn_layer: nn.Module, moe_layer: nn.Module, hidden_dim: int):
        super().__init__()
        self.attn = attn_layer
        self.moe = moe_layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

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

        # --- STAGE 1: Main Stream (Attn Chunk 1) ---
        nvtx.range_push("Stream 1: Attn(Chunk1) [COMPUTE]")
        residual_1 = x1
        x1_mid = residual_1 + self.attn(self.norm1(x1))
        moe_in_1 = self.norm2(x1_mid)
        nvtx.range_pop()

        # Signal ready
        self.moe_in_ready.record(torch.cuda.current_stream())

        # --- STAGE 2: Overlap ---

        # A. SIDE STREAM (MoE Chunk 1)
        with torch.cuda.stream(self.moe_stream):
            self.moe_stream.wait_event(self.moe_in_ready)

            nvtx.range_push("Stream 2: MoE(Chunk1) [COMM/EXP]")
            moe_out_1 = self.moe(moe_in_1)
            nvtx.range_pop()

            self.moe_out_done.record(self.moe_stream)
            x1_final = x1_mid + moe_out_1

        # B. MAIN STREAM (Attn Chunk 2)
        nvtx.range_push("Stream 1: Attn(Chunk2) [COMPUTE]")
        residual_2 = x2
        x2_mid = residual_2 + self.attn(self.norm1(x2))
        moe_in_2 = self.norm2(x2_mid)
        nvtx.range_pop()

        # --- STAGE 3: Join ---
        torch.cuda.current_stream().wait_event(self.moe_out_done)

        nvtx.range_push("Stream 1: MoE(Chunk2) [COMPUTE]")
        moe_out_2 = self.moe(moe_in_2)
        x2_final = x2_mid + moe_out_2
        nvtx.range_pop()

        return torch.cat([x1_final, x2_final], dim=0)

    def _forward_sequential(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        x = resid + self.attn(self.norm1(x))
        resid = x
        x = resid + self.moe(self.norm2(x))
        return x
