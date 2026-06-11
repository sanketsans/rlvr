"""Weights & Biases logging helpers."""

from __future__ import annotations

import os
from typing import List, Optional

import wandb

from qwen3_rlvr.env import load_project_env
from qwen3_rlvr.eval.pass_at_k import PassAtKResult


def log_pass_at_k_to_wandb(
    result: PassAtKResult,
    project: str,
    name: str,
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> None:
    load_project_env()
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is not set. Add it to qwen3_rlvr/.env or export it in your shell."
        )

    config = result.to_summary_dict()
    run = wandb.init(project=project, name=name, entity=entity, tags=tags, config=config)

    for key, value in result.pass_at_k.items():
        wandb.log({f"eval/{key}": value})
    wandb.log({"eval/avg_num_correct": result.avg_num_correct})

    table = wandb.Table(columns=["example_id", "question", "ground_truth", "num_correct", "first_completion"])
    for row in result.per_question[: min(50, len(result.per_question))]:
        table.add_data(
            row.example_id,
            row.question[:200],
            row.ground_truth,
            row.num_correct,
            row.completions[0][:500] if row.completions else "",
        )
    wandb.log({"eval/sample_predictions": table})
    run.finish()
