#!/usr/bin/env python3
"""Supervised fine-tuning on curated rejection-sampling data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.env import load_project_env
from qwen3_rlvr.logging import setup_logger
from qwen3_rlvr.logging.resource_monitor import ResourceMonitor
from qwen3_rlvr.sft.trainer import CurriculumConfig, SFTConfig, SFTTrainer

logger = setup_logger(__name__)

load_project_env()


def _load_config(path: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _apply_overrides(cfg: dict, overrides: List[str]) -> dict:
    if not overrides:
        return cfg
    merged = OmegaConf.create(cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        OmegaConf.update(merged, key, value, merge=False)
    return OmegaConf.to_container(merged, resolve=True)


def _get(cfg: dict, *keys: str, default: Any = None) -> Any:
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _build_curriculum_config(dataset_cfg: dict) -> Optional[CurriculumConfig]:
    curriculum_cfg = dataset_cfg.get("curriculum")
    if not curriculum_cfg:
        return None
    return CurriculumConfig(
        enabled=bool(curriculum_cfg.get("enabled", False)),
        processed_prompt_ids_path=(
            curriculum_cfg.get("processed_prompt_ids")
            or curriculum_cfg.get("processed_prompt_ids_path")
        ),
        steps_per_phase=curriculum_cfg.get("steps_per_phase"),
        phases=curriculum_cfg.get(
            "phases",
            [
                {"easy": 0.8, "mid": 0.2, "hard": 0.0, "very_hard": 0.0},
                {"easy": 0.4, "mid": 0.4, "hard": 0.1, "very_hard": 0.1},
                {"easy": 0.1, "mid": 0.5, "hard": 0.25, "very_hard": 0.15},
            ],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT on rejection-sampling curated data.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--override", action="append", default=[], help="key=value OmegaConf override"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = _apply_overrides(_load_config(args.config), args.override)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    dataset_cfg = _get(cfg, "dataset", default={}) or {}
    sft_cfg = _get(cfg, "sft", default={}) or {}
    logging_cfg = _get(cfg, "logging", default={}) or {}
    wandb_cfg = _get(logging_cfg, "wandb", default={}) or {}
    curriculum_cfg = _build_curriculum_config(dataset_cfg)
    eval_cfg = _get(cfg, "eval", default={}) or {}

    processed_prompt_ids = dataset_cfg.get("processed_prompt_ids") or dataset_cfg.get(
        "processed_prompt_ids_path"
    )

    trainer_cfg = SFTConfig(
        model_path=cfg["model"],
        output_dir=cfg["output_dir"],
        manifest_path=dataset_cfg.get("manifest"),
        jsonl_path=dataset_cfg.get("jsonl"),
        variant=dataset_cfg.get("variant", "top2"),
        processed_prompt_ids_path=processed_prompt_ids,
        curriculum=curriculum_cfg,
        include_original=dataset_cfg.get("include_original", True),
        max_samples=dataset_cfg.get("max_samples"),
        max_steps=args.max_steps or sft_cfg.get("max_steps", 500),
        batch_size=args.batch_size or sft_cfg.get("batch_size", 4),
        grad_accum_steps=args.grad_accum_steps or sft_cfg.get("grad_accum_steps", 4),
        lr=args.lr if args.lr is not None else sft_cfg.get("lr", 2e-5),
        lr_scheduler=sft_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.03),
        min_lr_ratio=sft_cfg.get("min_lr_ratio", 0.1),
        max_seq_length=sft_cfg.get("max_seq_length", 2048),
        grad_clip=sft_cfg.get("grad_clip", 1.0),
        dtype=sft_cfg.get("dtype", "bfloat16"),
        seed=cfg.get("seed", 42),
        log_every_steps=logging_cfg.get("log_every_steps", sft_cfg.get("log_every_steps", 10)),
        save_every_steps=sft_cfg.get("save_every_steps", 100),
        problem_weighted_sampling=sft_cfg.get("problem_weighted_sampling", True),
        wandb_project=None if args.no_wandb else wandb_cfg.get("project"),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_name=args.wandb_name or wandb_cfg.get("name"),
        wandb_tags=(
            [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            if args.wandb_tags
            else wandb_cfg.get("tags")
        ),
        eval_every_steps=eval_cfg.get("every_steps", 25),
        eval_recipes=eval_cfg.get("recipes", ["gsm8k_test"]),
        eval_primary_recipe=eval_cfg.get("primary_recipe", "gsm8k_test"),
        eval_max_samples=eval_cfg.get("max_samples", 200),
        eval_max_new_tokens=eval_cfg.get("max_new_tokens", 512),
        eval_n_generations=eval_cfg.get("n_generations", 5),
        eval_k=eval_cfg.get("k", [1, 3, 5]),
        eval_temperature=eval_cfg.get("temperature", 0.7),
        eval_question_batch_size=eval_cfg.get("question_batch_size", 8),
    )

    logger.info(f"SFT training: {trainer_cfg.model_path}")
    logger.info(f"Variant: {trainer_cfg.variant}, examples cap: {trainer_cfg.max_samples}")
    logger.info(f"Include original GSM8K: {trainer_cfg.include_original}")
    if curriculum_cfg and curriculum_cfg.enabled:
        logger.info(f"Curriculum enabled, steps/phase: {curriculum_cfg.steps_per_phase or 'auto'}")
    logger.info(f"Output: {trainer_cfg.output_dir}")
    with ResourceMonitor(
        trainer_cfg.output_dir + "/resource_monitor.json", interval_s=2.0, label="sft"
    ) as monitor:
        SFTTrainer(trainer_cfg).train()
    monitor.print_summary()


if __name__ == "__main__":
    main()
