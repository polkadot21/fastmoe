import torch
import torch.nn as nn
from torch.profiler import record_function


class PipelinedMoEBlock(nn.Module):
    def __init__(self, block_module, comm_stream, num_chunks=2):
        super().__init__()
        self.attn = block_module.attn
        self.norm1 = block_module.norm1
        self.norm2 = block_module.norm2
        self.moe = block_module.ff

        # STREAM 1: Communication (NCCL Dispatch/Combine)
        self.comm_stream = comm_stream

        # STREAM 2: Expert Compute (Low Priority)
        # We give it lower priority so Attention (Main Stream) always preempts it.
        # This ensures the "next chunk" is prepared as fast as possible.
        self.expert_stream = torch.cuda.Stream(priority=-1)

        self.default_num_chunks = num_chunks

    def forward(self, x):
        """
        3-Stream Pipeline for Maximum Overlap:
        Stream A (Main):   Attention(i) -> Gate(i)
        Stream B (Comm):   Dispatch(i)  -> Combine(i)
        Stream C (Expert): Experts(i)

        Overlap Targets:
        1. Experts(i)   || Attention(i+1) (Compute vs Compute)
        2. Experts(i)   || Dispatch(i+1)  (Compute vs Comm)
        """
        # Dynamic chunking for variable batch sizes
        num_chunks = self.default_num_chunks
        if x.shape[0] < num_chunks:
            num_chunks = 1

        chunks = x.chunk(num_chunks, dim=0)
        actual_num_chunks = len(chunks)

        # --- State Tracking ---
        chunk_meta = [{} for _ in range(actual_num_chunks)]
        final_results = [None] * actual_num_chunks

        # Events for dependency management
        events_gate_ready = [torch.cuda.Event() for _ in range(actual_num_chunks)]
        events_dispatch_done = [torch.cuda.Event() for _ in range(actual_num_chunks)]
        events_expert_done = [torch.cuda.Event() for _ in range(actual_num_chunks)]
        events_combine_done = [torch.cuda.Event() for _ in range(actual_num_chunks)]

        # Keep async handles alive
        dispatch_handles = [None] * actual_num_chunks
        combine_handles = [None] * actual_num_chunks

        main_stream = torch.cuda.current_stream()

        for i, x_chunk in enumerate(chunks):
            # -----------------------------------------------------------------
            # 1. ATTENTION + GATING (Main Stream)
            # -----------------------------------------------------------------
            with record_function(f"Comp: Attn+Gate [{i}]"):
                h = self.norm1(x_chunk)
                h = self.attn(h)
                x_resid = x_chunk + h
                moe_input = self.norm2(x_resid)

                perm, counts, rev, w, s = self.moe.gate_and_sort(moe_input)

                chunk_meta[i] = {
                    "resid": x_resid,
                    "perm": perm,
                    "counts": counts,
                    "rev": rev,
                    "w": w,
                    "s": s,
                }
                del h, moe_input  # Early free

            events_gate_ready[i].record(main_stream)

            # -----------------------------------------------------------------
            # 2. DISPATCH (Comm Stream)
            # -----------------------------------------------------------------
            with torch.cuda.stream(self.comm_stream):
                # Must wait for Gating data
                self.comm_stream.wait_event(events_gate_ready[i])

                with record_function(f"Comm: Dispatch Async [{i}]"):
                    rd, handle, meta = self.moe.dispatch_exchange_async(
                        chunk_meta[i]["perm"], chunk_meta[i]["counts"]
                    )
                    chunk_meta[i]["rd"] = rd
                    chunk_meta[i]["meta_dispatch"] = meta
                    dispatch_handles[i] = handle

                    # Cleanup
                    chunk_meta[i]["perm"] = None
                    chunk_meta[i]["counts"] = None

            events_dispatch_done[i].record(self.comm_stream)

            # -----------------------------------------------------------------
            # 3. EXPERTS (Expert Stream)
            # -----------------------------------------------------------------
            # Overlaps with: Attn(i+1) on Main AND Dispatch(i+1) on Comm
            with torch.cuda.stream(self.expert_stream):
                # Must wait for Dispatch to provide data
                self.expert_stream.wait_event(events_dispatch_done[i])

                with record_function(f"Comp: Experts [{i}]"):
                    recv_counts, send_splits = chunk_meta[i]["meta_dispatch"]

                    eo = self.moe.compute_experts(chunk_meta[i]["rd"], recv_counts)
                    chunk_meta[i]["eo"] = eo

                    chunk_meta[i]["rd"] = None  # Free large recv buffer

            events_expert_done[i].record(self.expert_stream)

            # -----------------------------------------------------------------
            # 4. COMBINE (Comm Stream)
            # -----------------------------------------------------------------
            with torch.cuda.stream(self.comm_stream):
                # Must wait for Experts to finish computing
                self.comm_stream.wait_event(events_expert_done[i])

                with record_function(f"Comm: Combine Async [{i}]"):
                    fd, handle = self.moe.combine_exchange_async(
                        chunk_meta[i]["eo"],
                        chunk_meta[i]["meta_dispatch"][0],  # recv_counts
                        chunk_meta[i]["meta_dispatch"][1],  # send_splits
                    )
                    chunk_meta[i]["fd"] = fd
                    combine_handles[i] = handle

                    chunk_meta[i]["eo"] = None

            events_combine_done[i].record(self.comm_stream)

        # -----------------------------------------------------------------
        # FINALIZE (Main Stream)
        # -----------------------------------------------------------------
        for i in range(actual_num_chunks):
            # We only need to wait for the final Combine step of this chunk
            main_stream.wait_event(events_combine_done[i])

            # Ensure CPU handles are cleared (for safety)
            if dispatch_handles[i]:
                dispatch_handles[i].wait()
            if combine_handles[i]:
                combine_handles[i].wait()

            m = chunk_meta[i]
            res = self.moe.unpermute(m["fd"], m["rev"], m["w"], m["s"])
            final_results[i] = res + m["resid"]

            # Help GC
            chunk_meta[i] = None

        return torch.cat(final_results, dim=0)
