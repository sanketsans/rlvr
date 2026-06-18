from qwen3_rlvr.rl.grpo import GRPOBatch, compute_advantages, compute_policy_loss

__all__ = [
    "GRPOBatch",
    "GRPOTrainer",
    "Trainer",
    "TrainerConfig",
    "compute_advantages",
    "compute_policy_loss",
    "ReinforceTrainer",
]


def __getattr__(name: str):
    if name in {"GRPOTrainer", "Trainer", "TrainerConfig", "ReinforceTrainer"}:
        from qwen3_rlvr.rl.trainer import GRPOTrainer, ReinforceTrainer, Trainer, TrainerConfig

        exports = {
            "GRPOTrainer": GRPOTrainer,
            "Trainer": Trainer,
            "TrainerConfig": TrainerConfig,
            "ReinforceTrainer": ReinforceTrainer,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
