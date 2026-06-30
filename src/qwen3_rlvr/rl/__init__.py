from qwen3_rlvr.rl.grpo import GRPOBatch, compute_advantages, compute_policy_loss
from qwen3_rlvr.rl.mopd import (
    MOPDBatch,
    combine_teacher_logps,
    compute_mopd_loss,
    compute_token_advantages,
    score_with_teachers,
)

__all__ = [
    "GRPOBatch",
    "GRPOTrainer",
    "Trainer",
    "TrainerConfig",
    "compute_advantages",
    "compute_policy_loss",
    "ReinforceTrainer",
    "MOPDBatch",
    "MOPDConfig",
    "MOPDTrainer",
    "combine_teacher_logps",
    "compute_mopd_loss",
    "compute_token_advantages",
    "score_with_teachers",
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
    if name in {"MOPDConfig", "MOPDTrainer"}:
        from qwen3_rlvr.rl.mopd import MOPDConfig, MOPDTrainer

        return {"MOPDConfig": MOPDConfig, "MOPDTrainer": MOPDTrainer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
