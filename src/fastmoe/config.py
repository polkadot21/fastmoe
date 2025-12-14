import functools
import os
import sys
from enum import Enum

import torch
import torch.distributed as dist
from loguru import logger
from pydantic_settings import BaseSettings


class MoEScale(str, Enum):
    DEBUG = "debug"
    GIGACHAT_10B = "gigachat-10b"
    GIGACHAT_700B = "gigachat-700b"
    TRACE_OPTIMIZED = "trace-optimized"


class MoESetup(BaseSettings):
    scale: MoEScale = MoEScale.DEBUG

    # Dimensions
    # Note: Global Batch Size in training is huge, but Micro-Batch per GPU is usually small (1-4).
    # We use a realistic per-GPU micro-batch here.
    batch_size: int = 4
    seq_len: int = 4096

    # Model Architecture
    hidden_dim: int = 4096

    # MoE Specifics
    num_experts: int = 8
    top_k: int = 2

    # Experiment Settings
    warmup_steps: int = 10
    active_steps: int = 50

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.seq_len * self.top_k

    @property
    def activation_volume_mb(self) -> float:
        """Size of the tensor being recombined in MB (Float32)"""
        # B * T * TopK * Hidden * 4 bytes
        return (self.total_tokens * self.hidden_dim * 4) / (1024**2)


class Config(BaseSettings):
    moe: MoESetup = MoESetup()
    logs_dir: str = "logs"
    log_level: str = "INFO"

    class Config:
        extra = "ignore"


def get_config(scale: MoEScale = MoEScale.DEBUG) -> MoESetup:
    """Factory for GigaChat MoE configurations"""

    if scale == MoEScale.DEBUG:
        return MoESetup(
            scale=MoEScale.DEBUG, batch_size=2, seq_len=128, hidden_dim=512, num_experts=4, top_k=2
        )

    elif scale == MoEScale.TRACE_OPTIMIZED:
        # Designed for 2 GPUs to show perfect overlap.
        # High Hidden Dim = Heavy Math.
        # High Batch = Heavy Payload.
        # Moderate Seq Len = Safety from OOM.
        return MoESetup(
            scale=MoEScale.TRACE_OPTIMIZED,
            batch_size=128,
            seq_len=512,
            hidden_dim=8192,  # Standard Llama-7B width. Heavy matrices.
            num_experts=8,  # 4 experts per GPU
            top_k=2,
        )

    elif scale == MoEScale.GIGACHAT_10B:
        # Configuration roughly matching Mixtral 8x7B or similar mid-sized MoE
        return MoESetup(
            scale=MoEScale.GIGACHAT_10B,
            batch_size=4,  # Micro-batch per GPU
            seq_len=4096,  # Standard context
            hidden_dim=4096,  # Standard 7B-10B model width
            num_experts=8,  # 8 Experts
            top_k=2,
        )

    elif scale == MoEScale.GIGACHAT_700B:
        # Configuration for massive scale (e.g., DeepSeek-V3, GPT-4 proxies)
        # The hidden dimension grows significantly, and expert count often increases.
        return MoESetup(
            scale=MoEScale.GIGACHAT_700B,
            batch_size=2,  # Reduced due to memory pressure
            seq_len=4096,
            hidden_dim=16384,  # Massive width
            num_experts=256,  # Extreme fragmentation
            top_k=4,  # Higher Top-K common at this scale
        )

    return MoESetup()


@functools.lru_cache
def init_app() -> Config:
    # 1. Load Configuration
    cfg = Config()

    # 2. Configure Logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=cfg.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",  # noqa
    )

    # 3. Initialize Distributed Environment (The new part)
    # torchrun sets these variables automatically
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            # NCCL is standard for GPUs, Gloo for CPU tests
            backend = "nccl" if torch.cuda.is_available() else "gloo"

            dist.init_process_group(backend=backend)

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            rank = dist.get_rank()
            if rank == 0:
                logger.info(
                    f"Distributed Init: Success. Backend={backend}, World={dist.get_world_size()}"
                )
        except Exception as e:
            logger.error(f"Distributed Init Failed: {e}")
            raise e
    else:
        logger.info(
            "No distributed environment detected (RANK/WORLD_SIZE missing). Running in single-process mode."  # noqa
        )

    return cfg
