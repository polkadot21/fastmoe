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

        # STREAM 1: Compute (Attention) - Replaces Default Stream
        self.comp_stream = torch.cuda.Stream()

        # STREAM 2: Communication
        self.comm_stream = comm_stream

        # STREAM 3: Expert Compute
        self.expert_stream = torch.cuda.Stream()

        self.default_num_chunks = num_chunks

    def forward(self, x):
        """
        True 3-Stream Overlap with Dedicated Streams
        """
        num_chunks = self.default_num_chunks
        if x.shape[0] < num_chunks:
            num_chunks = 1

        chunks = x.chunk(num_chunks, dim=0)
        actual_num_chunks = len(chunks)

        chunk_meta = [{} for _ in range(actual_num_chunks)]
        final_results = [None] * actual_num_chunks

        events_gate_ready = [torch.cuda.Event() for _ in range(actual_num_chunks)]
        events_dispatch_done = [torch.cuda.Event() for _ in range(actual_num_chunks)]
        events_expert_done = [torch.cuda.Event() for _ in range(actual_num_chunks)]
        events_combine_done = [torch.cuda.Event() for _ in range(actual_num_chunks)]

        dispatch_handles = [None] * actual_num_chunks
        combine_handles = [None] * actual_num_chunks

        # We must sync the incoming data (Default Stream) with our Comp Stream
        # to ensure 'chunks' are ready to be read.
        self.comp_stream.wait_stream(torch.cuda.current_stream())

        for i, x_chunk in enumerate(chunks):
            # -----------------------------------------------------------------
            # 1. ATTENTION (Comp Stream)
            # -----------------------------------------------------------------
            with torch.cuda.stream(self.comp_stream):
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

                events_gate_ready[i].record(self.comp_stream)

            # -----------------------------------------------------------------
            # 2. DISPATCH (Comm Stream)
            # -----------------------------------------------------------------
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(events_gate_ready[i])

                with record_function(f"Comm: Dispatch Async [{i}]"):
                    # This function now performs the CPU sync for metadata
                    # But since Attn[i+1] is on a different stream, CPU can submit it fast
                    rd, handle, meta = self.moe.dispatch_exchange_async(
                        chunk_meta[i]["perm"], chunk_meta[i]["counts"]
                    )
                    chunk_meta[i]["rd"] = rd
                    chunk_meta[i]["meta_dispatch"] = meta
                    dispatch_handles[i] = handle

                    chunk_meta[i]["perm"] = None
                    chunk_meta[i]["counts"] = None

                events_dispatch_done[i].record(self.comm_stream)

            # -----------------------------------------------------------------
            # 3. EXPERTS (Expert Stream)
            # -----------------------------------------------------------------
            with torch.cuda.stream(self.expert_stream):
                self.expert_stream.wait_event(events_dispatch_done[i])

                # Check CPU handle (Safety for Async NCCL)
                # Ideally, we skip this if we trust NCCL stream ordering
                if dispatch_handles[i]:
                    pass

                with record_function(f"Comp: Experts [{i}]"):
                    # Meta is now a list, no .item() calls inside!
                    recv_counts_list, send_splits = chunk_meta[i]["meta_dispatch"]

                    eo = self.moe.compute_experts(chunk_meta[i]["rd"], recv_counts_list)
                    chunk_meta[i]["eo"] = eo
                    chunk_meta[i]["rd"] = None

                events_expert_done[i].record(self.expert_stream)

            # -----------------------------------------------------------------
            # 4. COMBINE (Comm Stream)
            # -----------------------------------------------------------------
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(events_expert_done[i])

                with record_function(f"Comm: Combine Async [{i}]"):
                    fd, handle = self.moe.combine_exchange_async(
                        chunk_meta[i]["eo"],
                        chunk_meta[i]["meta_dispatch"][0],
                        chunk_meta[i]["meta_dispatch"][1],
                    )
                    chunk_meta[i]["fd"] = fd
                    combine_handles[i] = handle
                    chunk_meta[i]["eo"] = None

                events_combine_done[i].record(self.comm_stream)

        # -----------------------------------------------------------------
        # FINALIZE (Comp Stream -> Default)
        # -----------------------------------------------------------------
        # We process final additions on Comp Stream, then sync Default Stream to it.
        for i in range(actual_num_chunks):
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

        # Sync Default Stream to our work
        torch.cuda.current_stream().wait_stream(self.comp_stream)

        return torch.cat(final_results, dim=0)
