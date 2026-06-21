from qwen3_rlvr.logging.logger import setup_logger
from qwen3_rlvr.logging.resource_monitor import ResourceMonitor, sample_resources

__all__ = [
    "ResourceMonitor",
    "SFTWandbLogger",
    "log_pass_at_k_to_wandb",
    "sample_resources",
    "setup_logger",
]


def __getattr__(name: str):
    if name == "log_pass_at_k_to_wandb":
        from qwen3_rlvr.logging.wandb_logger import log_pass_at_k_to_wandb

        return log_pass_at_k_to_wandb
    if name == "SFTWandbLogger":
        from qwen3_rlvr.logging.wandb_sft import SFTWandbLogger

        return SFTWandbLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
