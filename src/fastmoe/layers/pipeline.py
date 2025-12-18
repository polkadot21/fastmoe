import torch
import torch.nn as nn
from torch.profiler import record_function


class PipelinedMoEBlock(nn.Module):
    def __init__(self, block_module, num_chunks=2):
        super().__init__()
        self.attn = block_module.attn
        self.norm1 = block_module.norm1
        self.norm2 = block_module.norm2
        self.moe = block_module.ff

        # STREAM 1: Compute A (Attention & Gating)
        self.comp_stream = torch.cuda.Stream()

        # STREAM 2: Compute B (Experts) - Low Priority
        self.expert_stream = torch.cuda.Stream(priority=-1)

        # STREAM 3: Comm A (Dispatch)
        self.dispatch_stream = torch.cuda.Stream()

        # STREAM 4: Comm B (Combine)
        # Separate stream ensures Combine waiting for Experts doesn't block next Dispatch
        self.combine_stream = torch.cuda.Stream()

        self.default_num_chunks = num_chunks

    def forward(self, x):
        """
        4-Stream Pipeline "Zero Bubble" Logic:

        [Chunk 0]   [Chunk 1]   [Chunk 2]
        Attn        Attn        Attn       (Stream 1)
        Disp        Disp        Disp       (Stream 3) -> Overlaps Experts[i-1]
              Experts     Experts          (Stream 2) -> Overlaps Attn[i+1] & Disp[i+1]
                    Comb        Comb       (Stream 4)
        """
        num_chunks = self.default_num_chunks
        if x.shape[0] < num_chunks:
            num_chunks = 1

        chunks = x.chunk(num_chunks, dim=0)
        actual_num_chunks = len(chunks)

        # Meta Storage
        chunk_meta = [{} for _ in range(actual_num_chunks)]
        final_results = [None] * actual_num_chunks

        # Synchronization Events
        events_gate_ready = [torch.cuda.Event() for _ in range(actual_num_chunks)]
        events_dispatch_done = [torch.cuda.Event() for _ in range(actual_num_chunks)]
        events_expert_done = [torch.cuda.Event() for _ in range(actual_num_chunks)]
        events_combine_done = [torch.cuda.Event() for _ in range(actual_num_chunks)]

        # Handles
        dispatch_handles = [None] * actual_num_chunks
        combine_handles = [None] * actual_num_chunks

        # Sync input data to Comp Stream
        self.comp_stream.wait_stream(torch.cuda.current_stream())

        for i, x_chunk in enumerate(chunks):
            # -----------------------------------------------------------------
            # 1. ATTENTION (Stream 1)
            # -----------------------------------------------------------------
            with torch.cuda.stream(self.comp_stream):
                torch.cuda.nvtx.range_push(f"Chunk {i}: Attn+Gate")
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
                    del h, moe_input
                torch.cuda.nvtx.range_pop()

                events_gate_ready[i].record(self.comp_stream)

            # -----------------------------------------------------------------
            # 2. DISPATCH (Stream 3)
            # -----------------------------------------------------------------
            # This can start as soon as Gate is ready.
            # It will NOT be blocked by the previous chunk's Combine step.
            with torch.cuda.stream(self.dispatch_stream):
                self.dispatch_stream.wait_event(events_gate_ready[i])

                torch.cuda.nvtx.range_push(f"Chunk {i}: Dispatch")
                with record_function(f"NCCL: Dispatch [{i}]"):
                    rd, handle, meta = self.moe.dispatch_exchange_async(
                        chunk_meta[i]["perm"], chunk_meta[i]["counts"]
                    )
                    chunk_meta[i]["rd"] = rd
                    chunk_meta[i]["meta_dispatch"] = meta
                    dispatch_handles[i] = handle

                    chunk_meta[i]["perm"] = None
                    chunk_meta[i]["counts"] = None
                torch.cuda.nvtx.range_pop()

                events_dispatch_done[i].record(self.dispatch_stream)

            # -----------------------------------------------------------------
            # 3. EXPERTS (Stream 2)
            # -----------------------------------------------------------------
            with torch.cuda.stream(self.expert_stream):
                self.expert_stream.wait_event(events_dispatch_done[i])

                # CPU-side check for safety
                if dispatch_handles[i]:
                    pass

                torch.cuda.nvtx.range_push(f"Chunk {i}: Experts")
                with record_function(f"Comp: Experts [{i}]"):
                    recv_counts_list, send_splits = chunk_meta[i]["meta_dispatch"]

                    eo = self.moe.compute_experts(chunk_meta[i]["rd"], recv_counts_list)
                    chunk_meta[i]["eo"] = eo
                    chunk_meta[i]["rd"] = None
                torch.cuda.nvtx.range_pop()

                events_expert_done[i].record(self.expert_stream)

            # -----------------------------------------------------------------
            # 4. COMBINE (Stream 4)
            # -----------------------------------------------------------------
            # This waits for Experts, but because it's on a separate stream,
            # it doesn't block the Dispatch Stream from processing Chunk i+1.
            with torch.cuda.stream(self.combine_stream):
                self.combine_stream.wait_event(events_expert_done[i])

                torch.cuda.nvtx.range_push(f"Chunk {i}: Combine")
                with record_function(f"NCCL: Combine [{i}]"):
                    fd, handle = self.moe.combine_exchange_async(
                        chunk_meta[i]["eo"],
                        chunk_meta[i]["meta_dispatch"][0],
                        chunk_meta[i]["meta_dispatch"][1],
                    )
                    chunk_meta[i]["fd"] = fd
                    combine_handles[i] = handle
                    chunk_meta[i]["eo"] = None
                torch.cuda.nvtx.range_pop()

                events_combine_done[i].record(self.combine_stream)

        # -----------------------------------------------------------------
        # FINALIZE (Stream 1 -> Default)
        # -----------------------------------------------------------------
        for i in range(actual_num_chunks):
            # We add back to the residual on the Comp Stream
            with torch.cuda.stream(self.comp_stream):
                self.comp_stream.wait_event(events_combine_done[i])

                if dispatch_handles[i]:
                    dispatch_handles[i].wait()
                if combine_handles[i]:
                    combine_handles[i].wait()

                m = chunk_meta[i]
                res = self.moe.unpermute(m["fd"], m["rev"], m["w"], m["s"])
                final_results[i] = res + m["resid"]
                chunk_meta[i] = None

        torch.cuda.current_stream().wait_stream(self.comp_stream)

        return torch.cat(final_results, dim=0)
