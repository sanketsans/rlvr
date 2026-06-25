"""W&B logging for SFT training."""

from __future__ import annotations

import os
from typing import List, Optional

import wandb

from qwen3_rlvr.env import load_project_env


class SFTWandbLogger:
    def __init__(
        self,
        project: str,
        name: str,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ):
        load_project_env()
        if not os.getenv("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is not set. Add it to rlvr/.env")
        self.run = wandb.init(project=project, name=name, entity=entity, tags=tags, config=config)

    def log_train(self, metrics: dict, step: int) -> None:
        payload = {f"train/{k}": v for k, v in metrics.items() if k != "step"}
        wandb.log(payload, step=step)

    def log_dataset(self, metrics: dict) -> None:
        wandb.log({f"dataset/{k}": v for k, v in metrics.items()}, step=0)

    def log_eval(self, metrics: dict, step: int) -> None:
        payload = {f"eval/{k}": v for k, v in metrics.items()}
        wandb.log(payload, step=step)

    def finish(self) -> None:
        self.run.finish()
