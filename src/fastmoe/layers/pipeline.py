import torch
import torch.nn as nn
from torch.profiler import record_function


class PipelinedMoEBlock(nn.Module):
    def __init__(self, block_module, comm_stream):
        super().__init__()
        # 1. We share the layers from the original block
        self.attn = block_module.attn
        self.norm1 = block_module.norm1
        self.norm2 = block_module.norm2
        self.moe = block_module.ff

        self.comm_stream = comm_stream

    def forward(self, x):
        """
        Hybrid Pipeline:
        1. Full Batch Attention (Max Efficiency)
        2. Micro-Batch MoE (Comm/Comp Overlap)
        """
        B, T, D = x.shape

        # --- PHASE 1: STANDARD ATTENTION (Full Batch) ---
        # No splitting here. Keep GPU saturated.
        with record_function("Standard: Attention"):
            residual = x
            x = self.norm1(x)
            x = self.attn(x)
            x = x + residual

        # --- PHASE 2: PIPELINED MOE ---
        # Now we split because we need to hide the All-to-All latency
        moe_input = self.norm2(x)

        # Pre-allocate output tensor to avoid 'torch.cat' memory spike/copy
        final_out = torch.empty_like(x)

        # Split into 2 micro-batches
        # We use 'view' slicing to avoid copying data
        chunks = moe_input.chunk(2, dim=0)
        c1, c2 = chunks[0], chunks[1]

        # Slices for writing output
        out_c1 = final_out.narrow(0, 0, B // 2)
        out_c2 = final_out.narrow(0, B // 2, B // 2)

        # Metadata containers
        c1_meta = {}
        c2_meta = {}

        # Events
        ev_dispatch_c1_done = torch.cuda.Event()
        ev_compute_c1_done = torch.cuda.Event()
        ev_dispatch_c2_done = torch.cuda.Event()

        # --- STEP 1: Process C1 (Dispatch) ---
        # We do this on Main Stream to start the pipeline
        with record_function("Pipe: Gate C1"):
            perm1, sc1, rev1, w1, s1 = self.moe.gate_and_sort(c1)
            c1_meta = {"rev": rev1, "w": w1, "s": s1, "sc": sc1}

        # Comm Stream: Dispatch C1
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with record_function("Pipe: Dispatch C1"):
                rd1, rc1 = self.moe.dispatch_exchange(perm1, sc1)
                c1_meta["rd"] = rd1
                c1_meta["rc"] = rc1
            ev_dispatch_c1_done.record()

        # --- STEP 2: Process C2 (Gate) & Compute C1 ---

        # Main Stream: Gate C2 (Compute)
        # This runs WHILE Dispatch C1 is happening on Comm Stream
        with record_function("Pipe: Gate C2"):
            perm2, sc2, rev2, w2, s2 = self.moe.gate_and_sort(c2)
            c2_meta = {"rev": rev2, "w": w2, "s": s2, "sc": sc2}

        # Main Stream: Compute C1
        # Must wait for Dispatch C1
        torch.cuda.current_stream().wait_event(ev_dispatch_c1_done)
        with record_function("Pipe: Experts C1"):
            eo1 = self.moe.compute_experts(c1_meta["rd"], c1_meta["rc"])
            c1_meta["eo"] = eo1
        ev_compute_c1_done.record()

        # --- STEP 3: Dispatch C2 & Combine C1 ---

        # Comm Stream: Dispatch C2 || Combine C1
        with torch.cuda.stream(self.comm_stream):
            # Wait for Gate C2 to be ready
            self.comm_stream.wait_stream(torch.cuda.current_stream())

            with record_function("Pipe: Dispatch C2"):
                rd2, rc2 = self.moe.dispatch_exchange(perm2, sc2)
                c2_meta["rd"] = rd2
                c2_meta["rc"] = rc2
            ev_dispatch_c2_done.record()

            # Combine C1 (Reverse All-to-All)
            # Must wait for Experts C1
            self.comm_stream.wait_event(ev_compute_c1_done)
            with record_function("Pipe: Combine C1"):
                fd1 = self.moe.combine_exchange(c1_meta["eo"], c1_meta["rc"], c1_meta["sc"])
                # Unpermute can happen on Comm stream or Compute.
                # Let's do it here to save main stream cycles.
                res1 = self.moe.unpermute(fd1, c1_meta["rev"], c1_meta["w"], c1_meta["s"])
                # Write directly to pre-allocated output
                out_c1.copy_(res1)

        # --- STEP 4: Compute C2 ---

        # Main Stream: Compute Experts C2
        # Must wait for Dispatch C2
        torch.cuda.current_stream().wait_event(ev_dispatch_c2_done)
        with record_function("Pipe: Experts C2"):
            eo2 = self.moe.compute_experts(c2_meta["rd"], c2_meta["rc"])

        # --- STEP 5: Combine C2 ---

        # Finalize on Main Stream (or Comm, but we need to join anyway)
        # We need to make sure Comm stream is done with Combine C1 before we return
        torch.cuda.current_stream().wait_stream(self.comm_stream)

        with record_function("Pipe: Combine C2"):
            fd2 = self.moe.combine_exchange(eo2, c2_meta["rc"], c2_meta["sc"])
            res2 = self.moe.unpermute(fd2, c2_meta["rev"], c2_meta["w"], c2_meta["s"])
            out_c2.copy_(res2)

        # Add residual connection (x is already moe_input + residual from attn)
        # We need to be careful: 'moe_input' was x after norm.
        # We want: output = x + MoE(Norm(x))
        # 'final_out' currently contains MoE(Norm(x))

        return x + final_out
