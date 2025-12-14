import torch.nn as nn
import torch.nn.functional as F

from fastmoe import consts
from fastmoe.layers.moe import MoEFeedForward
from fastmoe.layers.pipelined_block import PipelinedMoEBlock
from fastmoe.streams import MoEStreamManager  # <--- New Import


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):  # x: [B,T,D]
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.hd).transpose(1, 2)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.hd**0.5)
        att = att.softmax(dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SequentialBlock(nn.Module):
    """
    Standard sequential execution:
    1. Attention
    2. MoE (or FF)
    No overlap.
    """

    def __init__(self, attn_layer, ff_layer, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_layer
        self.norm2 = nn.LayerNorm(dim)
        self.ff = ff_layer

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyModel(nn.Module):
    def __init__(
        self,
        in_dim=512,
        dim=512,
        n_heads=8,
        ff_dim=2048,
        n_layers=4,
        num_experts=4,
        implementation=consts.MoEImplementation.FAST,
        use_moe=True,
        top_k=2,  # Added explicit arg for clarity
    ):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim, bias=False)
        self.blocks = nn.ModuleList()
        self.out = nn.Linear(dim, in_dim, bias=False)

        # We hold the manager here, but init it lazily in .to()
        self.stream_manager = None

        for i in range(n_layers):
            attn = MultiheadSelfAttention(dim, n_heads)

            # Support alternating or full MoE
            layer_uses_moe = (
                use_moe  # Simpler logic: if use_moe=True, all layers use it for this demo
            )

            if layer_uses_moe:
                ff = MoEFeedForward(
                    dim, ff_dim, num_experts=num_experts, top_k=top_k, implementation=implementation
                )
            else:
                ff = FeedForward(dim, ff_dim)

            if layer_uses_moe and (
                implementation == consts.MoEImplementation.FAST or implementation == "fast"
            ):
                # We pass manager=None for now. It will be injected in .to()
                block = PipelinedMoEBlock(attn, ff, dim, manager=None, block_idx=i)
            else:
                block = SequentialBlock(attn, ff, dim)

            self.blocks.append(block)

    def to(self, *args, **kwargs):
        """
        Override .to() to detect when the model moves to GPU.
        When it does, we initialize the MoEStreamManager and inject it into all blocks.
        """
        super().to(*args, **kwargs)

        # Check if we are now on a CUDA device
        try:
            device = next(self.parameters()).device
        except StopIteration:
            return self

        if device.type == "cuda" and self.stream_manager is None:
            self.stream_manager = MoEStreamManager(device)

            count = 0
            for block in self.blocks:
                if isinstance(block, PipelinedMoEBlock):
                    block.manager = self.stream_manager
                    count += 1

        return self

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)
