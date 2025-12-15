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

        # Higher priority helps NCCL/comm progress when GEMMs saturate SMs.
        self.moe_stream = torch.cuda.Stream(device=device, priority=-1)

        self.moe_in_ready = torch.cuda.Event(enable_timing=False)
        self.moe_out_done = torch.cuda.Event(enable_timing=False)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            ranks = list(range(world_size))

            self.group_chunk1 = dist.new_group(ranks=ranks)
            self.group_chunk2 = dist.new_group(ranks=ranks)
        else:
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
