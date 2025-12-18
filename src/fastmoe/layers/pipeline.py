import torch
import torch.nn as nn
from torch.profiler import record_function

from fastmoe.models.tiny_model import Block


class PipelinedMoEBlock(nn.Module):
    def __init__(
        self, block_module: Block, comm_stream: torch.cuda.Stream, num_chunks: int
    ) -> None:
        super().__init__()

        self.attn = block_module.attn
        self.norm1 = block_module.norm1
        self.norm2 = block_module.norm2
        self.moe = block_module.ff
        self.comm_stream = comm_stream
        self.num_chunks = num_chunks

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

        # 1. Pre-Attention Chunking (Views, zero-copy)
        chunks = x.chunk(self.num_chunks, dim=0)

        # State tracking for the pipeline
        chunk_meta = [{} for _ in range(self.num_chunks)]

        # Events for synchronization
        # We need to know when a chunk is ready to Dispatch (after Gate)
        events_ready_to_dispatch = [torch.cuda.Event() for _ in range(self.num_chunks)]
        # We need to know when Dispatch is done so we can Compute Experts
        events_dispatch_done = [torch.cuda.Event() for _ in range(self.num_chunks)]
        # We need to know when Experts are done so we can Combine
        events_experts_done = [torch.cuda.Event() for _ in range(self.num_chunks)]

        # Output buffer
        final_results = [None] * self.num_chunks

        current_stream = torch.cuda.current_stream()

        # --- LOOP: FILL THE PIPELINE ---
        for i, x_chunk in enumerate(chunks):
            # === COMPUTE STREAM: ATTENTION & GATING ===
            # This runs immediately for C(i). If i > 0, this overlaps with Dispatch(i-1)
            with record_function(f"Comp: Attn+Gate [{i}]"):
                # 1. Standard Attention Block
                h = self.norm1(x_chunk)
                h = self.attn(h)
                x_resid = x_chunk + h  # Residual 1

                # 2. MoE Setup
                moe_input = self.norm2(x_resid)

                # 3. Gate
                perm, sc, rev, w, s = self.moe.gate_and_sort(moe_input)

                # Store metadata for later stages
                chunk_meta[i] = {
                    "resid": x_resid,
                    "perm": perm,
                    "sc": sc,
                    "rev": rev,
                    "w": w,
                    "s": s,
                }

                # Clean up intermediate tensors to reduce peak memory
                del h, moe_input

            # Signal that C(i) is ready for Dispatch
            events_ready_to_dispatch[i].record(current_stream)

            # === COMM STREAM: DISPATCH ===
            with torch.cuda.stream(self.comm_stream):
                # Wait for Gate(i) to finish
                self.comm_stream.wait_event(events_ready_to_dispatch[i])

                # Check previous Dispatch dependency?
                # Implicitly serial on comm_stream, so Dispatch(i) waits for Combine(i-1) automatically. # noqa

                with record_function(f"Comm: Dispatch [{i}]"):
                    rd, rc, sl = self.moe.dispatch_exchange(
                        chunk_meta[i]["perm"], chunk_meta[i]["sc"]
                    )
                    chunk_meta[i]["rd"] = rd
                    chunk_meta[i]["rc"] = rc
                    chunk_meta[i]["sl"] = sl

                    # Can free "perm" and "sc" now? Yes.
                    del chunk_meta[i]["perm"], chunk_meta[i]["sc"]

                # Signal that Dispatch(i) is done (Input for Experts available)
                events_dispatch_done[i].record(self.comm_stream)

            # === COMPUTE STREAM: EXPERTS (Delayed for previous chunk) ===
            # While we just queued Dispatch for C(i), we check if we can run Experts for C(i-1)
            # or even C(i) if we are waiting anyway.

            # To strictly follow "Attn C2 begins... All-to-all C1 start",
            # we simply process the Attention loop continuously.
            # But we must insert the Expert compute slots somewhere.
            # Strategy: Try to process Experts for ANY chunk that is ready.

            # Optimization: If this is not the first chunk, C(i-1) might be ready for Experts.
            prev_idx = i - 1
            if prev_idx >= 0:
                # Wait for Dispatch(i-1) to finish
                current_stream.wait_event(events_dispatch_done[prev_idx])

                with record_function(f"Comp: Experts [{prev_idx}]"):
                    meta_prev = chunk_meta[prev_idx]
                    eo = self.moe.compute_experts(
                        meta_prev["rd"]
                    )  # , meta_prev["rc"]) # Fixed signature call
                    meta_prev["eo"] = eo
                    # Free heavy receive buffer
                    del meta_prev["rd"]

                # Signal Experts(i-1) done
                events_experts_done[prev_idx].record(current_stream)

                # Schedule Combine(i-1) on Comm Stream
                with torch.cuda.stream(self.comm_stream):
                    self.comm_stream.wait_event(events_experts_done[prev_idx])

                    with record_function(f"Comm: Combine [{prev_idx}]"):
                        fd = self.moe.combine_exchange(
                            meta_prev["eo"], meta_prev["rc"], meta_prev["sl"]
                        )
                        meta_prev["fd"] = fd
                        # Free expert output and split info
                        del meta_prev["eo"], meta_prev["rc"], meta_prev["sl"]

        # === DRAIN PIPELINE ===
        # The loop finishes having processed Attn(Last).
        # We still need to run Experts(Last) and Combine(Last).
        last_idx = self.num_chunks - 1

        # 1. Experts Last
        current_stream.wait_event(events_dispatch_done[last_idx])
        with record_function(f"Comp: Experts [{last_idx}]"):
            meta_last = chunk_meta[last_idx]
            eo = self.moe.compute_experts(meta_last["rd"])
            meta_last["eo"] = eo
            del meta_last["rd"]
        events_experts_done[last_idx].record(current_stream)

        # 2. Combine Last
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(events_experts_done[last_idx])
            with record_function(f"Comm: Combine [{last_idx}]"):
                fd = self.moe.combine_exchange(meta_last["eo"], meta_last["rc"], meta_last["sl"])
                meta_last["fd"] = fd
                del meta_last["eo"], meta_last["rc"], meta_last["sl"]

        # === FINALIZE (Unpermute & Add Residual) ===
        # This is compute-bound but fast. We do this after combine is ready.
        # We synchronize streams at the very end or per chunk.

        # Ensure Comm stream is done with everything
        current_stream.wait_stream(self.comm_stream)

        for i in range(self.num_chunks):
            meta = chunk_meta[i]
            # Unpermute
            res = self.moe.unpermute(meta["fd"], meta["rev"], meta["w"], meta["s"])
            # Add Residual 2
            out = res + meta["resid"]
            final_results[i] = out

            # Aggressive cleanup
            del chunk_meta[i]

        return torch.cat(final_results, dim=0)
