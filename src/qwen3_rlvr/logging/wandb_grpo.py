"""W&B logging for GRPO training."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import wandb

from qwen3_rlvr.env import load_project_env


class GRPO_WandbLogger:
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

    def log_eval(self, metrics: dict, step: int) -> None:
        wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

    def log_samples(self, records: List[dict], step: int, stage: str) -> None:
        table = wandb.Table(
            columns=["stage", "example_id", "question", "ground_truth", "reward", "completion"]
        )
        for row in records:
            table.add_data(
                stage,
                row.get("example_id"),
                (row.get("question") or "")[:200],
                row.get("ground_truth"),
                row.get("reward"),
                (row.get("completion") or "")[:500],
            )
        wandb.log({f"samples/{stage}": table}, step=step)

    def finish(self) -> None:
        self.run.finish()
