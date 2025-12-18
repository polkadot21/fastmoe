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
        self.comm_stream = comm_stream
        self.default_num_chunks = num_chunks

    def forward(self, x):
        """
        Micro-Batch Pipelining with Pre-Attention Chunking.

        Strategy:
        Split input X into N chunks immediately.

        Timeline (Example N=2):
        [Comp] Attn(C0)->Gate(C0) | Attn(C1)->Gate(C1) | Exp(C0) | Exp(C1)
        [Comm] ...................| Disp(C0) ..........| Comb(C0)| Disp(C1) ...

        This maximizes overlap: The GPU Compute is calculating Attention for C(i+1)
        while the GPU Link is dispatching C(i).
        """

        # Determine chunks dynamically based on input batch size
        # If batch is too small, fallback to fewer chunks or 1
        num_chunks = self.default_num_chunks
        if x.shape[0] < num_chunks:
            num_chunks = 1

        chunks = x.chunk(num_chunks, dim=0)
        actual_num_chunks = len(chunks)

        # State Storage
        chunk_meta = [{} for _ in range(actual_num_chunks)]
        final_results = [None] * actual_num_chunks

        # Track async communication handles
        dispatch_handles = [None] * actual_num_chunks
        combine_handles = [None] * actual_num_chunks

        # --- PIPELINE LOOP ---
        for i, x_chunk in enumerate(chunks):
            # 1. ATTENTION (Compute Heavy) - Hides communication of previous chunk
            with record_function(f"Comp: Attn+Gate [{i}]"):
                h = self.norm1(x_chunk)
                h = self.attn(h)
                x_resid = x_chunk + h
                moe_input = self.norm2(x_resid)

                # Gate immediately after Attn
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

            # 2. DISPATCH PREV CHUNK (Comm Heavy)
            # We start Dispatch for [i] IMMEDIATELY after gating.
            with torch.cuda.stream(self.comm_stream):
                # Wait for Gating [i] to complete on compute stream
                self.comm_stream.wait_stream(torch.cuda.current_stream())

                with record_function(f"Comm: Dispatch Async [{i}]"):
                    rd, handle, meta = self.moe.dispatch_exchange_async(
                        chunk_meta[i]["perm"], chunk_meta[i]["counts"]
                    )
                    chunk_meta[i]["rd"] = rd
                    chunk_meta[i]["meta_dispatch"] = meta  # (counts, send_splits)
                    dispatch_handles[i] = handle

                    # We can clear these now to save memory
                    chunk_meta[i]["perm"] = None
                    chunk_meta[i]["counts"] = None

            # 3. COMPUTE EXPERTS PREV CHUNK
            prev_idx = i - 1
            if prev_idx >= 0:
                # We need Dispatch [i-1] to be done.
                if dispatch_handles[prev_idx]:
                    dispatch_handles[prev_idx].wait()  # CPU wait (non-blocking if NCCL is smart)

                # Ensure Comm stream is synced with Compute stream regarding the data buffer
                torch.cuda.current_stream().wait_stream(self.comm_stream)

                with record_function(f"Comp: Experts [{prev_idx}]"):
                    m_prev = chunk_meta[prev_idx]
                    recv_counts, send_splits = m_prev["meta_dispatch"]

                    eo = self.moe.compute_experts(m_prev["rd"], recv_counts)
                    m_prev["eo"] = eo
                    m_prev["rd"] = None  # Free memory

                # > Start Combine for Prev [prev_idx]
                with torch.cuda.stream(self.comm_stream):
                    # Comm stream must wait for Expert Compute to finish
                    self.comm_stream.wait_stream(torch.cuda.current_stream())

                    with record_function(f"Comm: Combine Async [{prev_idx}]"):
                        fd, handle = self.moe.combine_exchange_async(
                            m_prev["eo"], recv_counts, send_splits
                        )
                        m_prev["fd"] = fd
                        combine_handles[prev_idx] = handle
                        m_prev["eo"] = None

        # --- DRAIN PIPELINE (Last Chunk) ---
        last_idx = actual_num_chunks - 1

        # Wait for Dispatch Last
        if dispatch_handles[last_idx]:
            dispatch_handles[last_idx].wait()

        torch.cuda.current_stream().wait_stream(self.comm_stream)

        with record_function(f"Comp: Experts [{last_idx}]"):
            m_last = chunk_meta[last_idx]
            recv_counts, send_splits = m_last["meta_dispatch"]
            eo = self.moe.compute_experts(m_last["rd"], recv_counts)
            m_last["eo"] = eo
            m_last["rd"] = None

        # Combine Last
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with record_function(f"Comm: Combine Async [{last_idx}]"):
                fd, handle = self.moe.combine_exchange_async(m_last["eo"], recv_counts, send_splits)
                m_last["fd"] = fd
                combine_handles[last_idx] = handle
                m_last["eo"] = None

        # --- FINALIZE (Unpermute) ---
        # Wait for all combines to finish
        for i in range(actual_num_chunks):
            if combine_handles[i]:
                combine_handles[i].wait()

        # Ensure data is visible on compute stream
        torch.cuda.current_stream().wait_stream(self.comm_stream)

        for i in range(actual_num_chunks):
            m = chunk_meta[i]
            res = self.moe.unpermute(m["fd"], m["rev"], m["w"], m["s"])
            final_results[i] = res + m["resid"]
            chunk_meta[i] = None

        return torch.cat(final_results, dim=0)
