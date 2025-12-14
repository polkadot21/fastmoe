import torch
import torch.nn as nn


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
            # Event: Signals that Main Stream has finished preparing input for MoE
            self.moe_in_ready = torch.cuda.Event(enable_timing=False)
            # Event: Signals that Side Stream has finished MoE computation
            self.moe_out_done = torch.cuda.Event(enable_timing=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Fallback for small batches or CPU
        if not self.use_streams or B < 2:
            return self._forward_sequential(x)

        # 1. Split Batch
        x1, x2 = x.chunk(2, dim=0)

        # =================================================================
        # STAGE 1: Process Chunk 1 (Main Stream)
        # =================================================================
        # Compute Attn(B1) to get inputs ready for MoE
        residual_1 = x1
        x1_mid = residual_1 + self.attn(self.norm1(x1))
        moe_in_1 = self.norm2(x1_mid)

        # [SYNC POINT A]
        # Record that Main Stream is done with moe_in_1
        self.moe_in_ready.record(torch.cuda.current_stream())

        # =================================================================
        # STAGE 2: Overlap (Side Stream || Main Stream)
        # =================================================================

        # --- SIDE STREAM (MoE Chunk 1) ---
        with torch.cuda.stream(self.moe_stream):
            # Wait until Main Stream has written moe_in_1
            self.moe_stream.wait_event(self.moe_in_ready)

            # Heavy Comm + Compute
            moe_out_1 = self.moe(moe_in_1)

            # [SYNC POINT B]
            # Record that Side Stream is finished
            self.moe_out_done.record(self.moe_stream)

            # Combine residual for Chunk 1
            x1_final = x1_mid + moe_out_1

        # --- MAIN STREAM (Attn Chunk 2) ---
        # This runs simultaneously with the block above!
        residual_2 = x2
        x2_mid = residual_2 + self.attn(self.norm1(x2))
        moe_in_2 = self.norm2(x2_mid)

        # =================================================================
        # STAGE 3: Join
        # =================================================================

        # Finish Chunk 2 (Sequential Tail)
        moe_out_2 = self.moe(moe_in_2)
        x2_final = x2_mid + moe_out_2

        # [SYNC POINT C]
        # We cannot merge until Side Stream is definitely done.
        torch.cuda.current_stream().wait_event(self.moe_out_done)

        return torch.cat([x1_final, x2_final], dim=0)

    def _forward_sequential(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        x = resid + self.attn(self.norm1(x))
        resid = x
        x = resid + self.moe(self.norm2(x))
        return x
