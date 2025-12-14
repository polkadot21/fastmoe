import torch
import torch.distributed as dist


class MoEStreamManager:
    """
    Centralized manager for MoE streams, events, AND Process Groups.
    Creates two distinct communication lanes (process groups) to allow
    simultaneous All-to-All operations.
    """

    def __init__(self, device: torch.device):
        self.device = device

        # 1. Compute Streams
        self.moe_stream = torch.cuda.Stream(device=device)

        # 2. Events
        self.moe_in_ready = torch.cuda.Event(enable_timing=False)
        self.moe_out_done = torch.cuda.Event(enable_timing=False)

        # 3. Communication Groups (The "Dual Lanes")
        # Creating new_group with the same ranks MULTIPLE times creates
        # distinct NCCL communicators (hardware channels).

        # Get world size/ranks
        if dist.is_initialized():
            world_size = dist.get_world_size()
            ranks = list(range(world_size))

            # Lane 1: For Chunk 1 (Side Stream)
            self.group_chunk1 = dist.new_group(ranks=ranks)

            # Lane 2: For Chunk 2 (Main Stream)
            self.group_chunk2 = dist.new_group(ranks=ranks)
        else:
            # Fallback for single-gpu debug
            self.group_chunk1 = None
            self.group_chunk2 = None

    def wait_main(self):
        torch.cuda.current_stream().wait_event(self.moe_out_done)

    def wait_side(self):
        self.moe_stream.wait_event(self.moe_in_ready)

    def record_ready(self):
        self.moe_in_ready.record(torch.cuda.current_stream())

    def record_done(self):
        self.moe_out_done.record(self.moe_stream)
