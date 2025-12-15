import torch
import torch.nn as nn
from torch.profiler import record_function


class PipelinedMoEBlock(nn.Module):
    def __init__(self, block_module, comm_stream):
        super().__init__()
        # We wrap the existing Block
        self.attn = block_module.attn
        self.norm1 = block_module.norm1
        self.norm2 = block_module.norm2
        self.moe = block_module.ff

        self.comm_stream = comm_stream

    def forward(self, x):
        """
        Input x: [B, T, D]
        We assume B is divisible by 2 for the pipeline.
        """
        B, T, D = x.shape

        # 1. Split Micro-Batch
        chunks = x.chunk(2, dim=0)
        c1, c2 = chunks[0], chunks[1]

        # Storage for intermediate results
        c1_attn_out = None
        c2_attn_out = None

        # Events for synchronization
        ev_c1_attn_done = torch.cuda.Event()
        ev_c2_attn_done = torch.cuda.Event()

        ev_c1_dispatch_done = torch.cuda.Event()
        ev_c2_dispatch_done = torch.cuda.Event()

        ev_c1_compute_done = torch.cuda.Event()
        ev_c2_compute_done = torch.cuda.Event()

        # Metadata storage
        c1_meta = {}
        c2_meta = {}

        # === PIPELINE STAGE 1: Attention C1 ===
        with record_function("Pipe: Attn C1"):
            c1_resid = c1
            c1 = self.norm1(c1)
            c1 = self.attn(c1)
            c1 = c1 + c1_resid  # Add residual
            c1_attn_out = c1
            ev_c1_attn_done.record()

        # === PIPELINE STAGE 2: Attention C2 || Dispatch C1 ===

        # Stream 2 (Comm): Start working on C1 as soon as Attn C1 is done
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ev_c1_attn_done)
            with record_function("Pipe: Dispatch C1"):
                # 1. Norm & Gate
                moe_in = self.norm2(c1_attn_out)
                permuted, send_counts, rev_idx, weights, shape = self.moe.gate_and_sort(moe_in)
                c1_meta = {"rev": rev_idx, "w": weights, "s": shape, "sc": send_counts}

                # 2. All-to-All Dispatch
                recv_data, recv_splits = self.moe.dispatch_exchange(permuted, send_counts)
                c1_meta["rc"] = recv_splits
                c1_meta["rd"] = recv_data
            ev_c1_dispatch_done.record()

        # Stream 1 (Compute): Do Attn C2
        with record_function("Pipe: Attn C2"):
            c2_resid = c2
            c2 = self.norm1(c2)
            c2 = self.attn(c2)
            c2 = c2 + c2_resid
            c2_attn_out = c2
            ev_c2_attn_done.record()

        # === PIPELINE STAGE 3: Compute C1 || Dispatch C2 ===

        # Stream 2 (Comm): Start C2 Dispatch
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ev_c2_attn_done)
            with record_function("Pipe: Dispatch C2"):
                moe_in = self.norm2(c2_attn_out)
                permuted, send_counts, rev_idx, weights, shape = self.moe.gate_and_sort(moe_in)
                c2_meta = {"rev": rev_idx, "w": weights, "s": shape, "sc": send_counts}

                recv_data, recv_splits = self.moe.dispatch_exchange(permuted, send_counts)
                c2_meta["rc"] = recv_splits
                c2_meta["rd"] = recv_data
            ev_c2_dispatch_done.record()

        # Stream 1 (Compute): Do MoE Compute C1
        # Wait for data to arrive from Dispatch C1
        torch.cuda.current_stream().wait_event(ev_c1_dispatch_done)

        with record_function("Pipe: Experts C1"):
            expert_out = self.moe.compute_experts(c1_meta["rd"], c1_meta["rc"])
            c1_meta["eo"] = expert_out
        ev_c1_compute_done.record()

        # === PIPELINE STAGE 4: Compute C2 || Combine C1 ===

        # Stream 2 (Comm): Combine C1 (Reverse All-to-All)
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ev_c1_compute_done)
            with record_function("Pipe: Combine C1"):
                final_data = self.moe.combine_exchange(c1_meta["eo"], c1_meta["rc"], c1_meta["sc"])
                # Unpermute
                out_c1 = self.moe.unpermute(final_data, c1_meta["rev"], c1_meta["w"], c1_meta["s"])
                c1_meta["out"] = out_c1 + c1_attn_out  # Add Residual

        # Stream 1 (Compute): Experts C2
        torch.cuda.current_stream().wait_event(ev_c2_dispatch_done)
        with record_function("Pipe: Experts C2"):
            expert_out = self.moe.compute_experts(c2_meta["rd"], c2_meta["rc"])
            c2_meta["eo"] = expert_out
        ev_c2_compute_done.record()

        # === PIPELINE STAGE 5: Combine C2 ===

        # We can do this on main stream now as we need to join
        torch.cuda.current_stream().wait_event(ev_c2_compute_done)  # Ensure experts done
        # Also need to make sure stream2 is done with C1 before we return everything
        torch.cuda.current_stream().wait_stream(self.comm_stream)

        # Finish C2 on main stream
        with record_function("Pipe: Combine C2"):
            final_data = self.moe.combine_exchange(c2_meta["eo"], c2_meta["rc"], c2_meta["sc"])
            out_c2 = self.moe.unpermute(final_data, c2_meta["rev"], c2_meta["w"], c2_meta["s"])
            out_c2 = out_c2 + c2_attn_out

        return torch.cat([c1_meta["out"], out_c2], dim=0)
