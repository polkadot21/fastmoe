from abc import ABC, abstractmethod

import torch
import torch.distributed as dist


class Communicator(ABC):
    @abstractmethod
    def exchange_counts(self, local_counts: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def all_to_all(
        self, x: torch.Tensor, send_counts: list[int], recv_counts: list[int]
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def exchange_expert_histogram(self, local_hist: torch.Tensor) -> torch.Tensor:
        """
        Exchanges the detailed histogram of tokens per expert.
        Input: [WorldSize, NumLocalExperts] (How many I send to each expert on each rank)
        Output: [WorldSize, NumLocalExperts] (How many I receive for my experts from each rank)
        """
        pass


class PytorchCommunicator(Communicator):
    def __init__(self, group=None):
        self.group = group
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def exchange_counts(self, local_counts: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return local_counts
        recv_counts = torch.zeros_like(local_counts)
        dist.all_to_all_single(recv_counts, local_counts, group=self.group)
        return recv_counts

    def all_to_all(
        self, x: torch.Tensor, send_counts: list[int], recv_counts: list[int]
    ) -> torch.Tensor:
        if self.world_size == 1:
            return x
        output = torch.empty(sum(recv_counts), x.size(1), device=x.device, dtype=x.dtype)
        dist.all_to_all_single(
            output,
            x,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.group,
        )
        return output

    def exchange_expert_histogram(self, local_hist: torch.Tensor) -> torch.Tensor:
        """
        Input: [WorldSize, ExpertsPerRank]
        Output: [WorldSize, ExpertsPerRank]
        """
        if self.world_size == 1:
            return local_hist
        recv_hist = torch.zeros_like(local_hist)
        # We use all_to_all_single directly as the tensor is contiguous and small
        dist.all_to_all_single(recv_hist, local_hist, group=self.group)
        return recv_hist


class FakeCommunicator(Communicator):
    """For CPU Unit Tests"""

    def exchange_counts(self, local_counts):
        return local_counts

    def all_to_all(self, x, send_counts, recv_counts):
        return x
