import torch


class MoEStreamManager:
    """
    Centralized manager for MoE streams and events.
    Ensures we only create ONE side stream for the entire model,
    rather than one per layer.
    """

    def __init__(self, device: torch.device):
        self.device = device

        # The Side Stream (for Comm + Expert)
        self.moe_stream = torch.cuda.Stream(device=device)

        # Reusable Synchronization Events
        # We can reuse these because Layer N finishes joining before Layer N+1 starts.
        self.moe_in_ready = torch.cuda.Event(enable_timing=False)
        self.moe_out_done = torch.cuda.Event(enable_timing=False)

    def wait_main(self):
        """Current stream waits for Side Stream (Join)"""
        torch.cuda.current_stream().wait_event(self.moe_out_done)

    def wait_side(self):
        """Side stream waits for Main Stream (Fork)"""
        self.moe_stream.wait_event(self.moe_in_ready)

    def record_ready(self):
        """Record that Main Stream has prepared input data"""
        self.moe_in_ready.record(torch.cuda.current_stream())

    def record_done(self):
        """Record that Side Stream has finished processing"""
        self.moe_out_done.record(self.moe_stream)
