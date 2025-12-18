import torch
import torch.nn as nn
from torch.profiler import record_function


class PipelinedMoEBlock(nn.Module):
    def __init__(self, block_module, streams, num_chunks=2, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx

        self.attn = block_module.attn
        self.norm1 = block_module.norm1
        self.norm2 = block_module.norm2
        self.moe = block_module.ff

        self.comp_stream = streams[0]
        self.dispatch_stream = streams[1]
        self.combine_stream = streams[2]
        self.expert_stream = streams[3]

        self.default_num_chunks = num_chunks

    def forward(self, x):
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

        self.comp_stream.wait_stream(torch.cuda.current_stream())

        for i, x_chunk in enumerate(chunks):
            # Explicit Tag for profiling
            tag = f"L{self.layer_idx}_C{i}"

            # 1. ATTENTION
            with torch.cuda.stream(self.comp_stream):
                # NVTX Range for tool visualization
                torch.cuda.nvtx.range_push(f"{tag}: Attn")
                with record_function(f"{tag}: Attn+Gate"):
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

            # 2. DISPATCH
            with torch.cuda.stream(self.dispatch_stream):
                self.dispatch_stream.wait_event(events_gate_ready[i])

                torch.cuda.nvtx.range_push(f"{tag}: Dispatch")
                # We do NOT use record_function here because we push it down to moe.py
                # This ensures the NCCL call is tightly wrapped.
                rd, handle, meta = self.moe.dispatch_exchange_async(
                    chunk_meta[i]["perm"],
                    chunk_meta[i]["counts"],
                    tag=f"{tag}_Dispatch",  # <--- Passing the tag down
                )
                chunk_meta[i]["rd"] = rd
                chunk_meta[i]["meta_dispatch"] = meta
                dispatch_handles[i] = handle

                chunk_meta[i]["perm"] = None
                chunk_meta[i]["counts"] = None
                torch.cuda.nvtx.range_pop()
                events_dispatch_done[i].record(self.dispatch_stream)

            # 3. EXPERTS
            with torch.cuda.stream(self.expert_stream):
                self.expert_stream.wait_event(events_dispatch_done[i])
                if dispatch_handles[i]:
                    pass

                torch.cuda.nvtx.range_push(f"{tag}: Experts")
                with record_function(f"{tag}: Experts"):
                    recv_counts_list, send_splits = chunk_meta[i]["meta_dispatch"]
                    eo = self.moe.compute_experts(chunk_meta[i]["rd"], recv_counts_list)
                    chunk_meta[i]["eo"] = eo
                    chunk_meta[i]["rd"] = None
                torch.cuda.nvtx.range_pop()
                events_expert_done[i].record(self.expert_stream)

            # 4. COMBINE
            with torch.cuda.stream(self.combine_stream):
                self.combine_stream.wait_event(events_expert_done[i])

                torch.cuda.nvtx.range_push(f"{tag}: Combine")
                # Pass tag down to moe.py
                fd, handle = self.moe.combine_exchange_async(
                    chunk_meta[i]["eo"],
                    chunk_meta[i]["meta_dispatch"][0],
                    chunk_meta[i]["meta_dispatch"][1],
                    tag=f"{tag}_Combine",  # <--- Passing the tag down
                )
                chunk_meta[i]["fd"] = fd
                combine_handles[i] = handle
                chunk_meta[i]["eo"] = None
                torch.cuda.nvtx.range_pop()
                events_combine_done[i].record(self.combine_stream)

        # FINALIZE
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

        torch.cuda.current_stream().wait_stream(self.comp_stream)
        return torch.cat(final_results, dim=0)
