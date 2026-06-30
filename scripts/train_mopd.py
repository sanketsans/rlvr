#!/usr/bin/env python3
"""Phase 3: MOPD (Multi-teacher On-Policy Distillation) on GSM8K.

Student samples its own rollouts; frozen teacher(s) score the same tokens to
produce a dense per-token reverse-KL advantage (sg[log π_T − log π_prox]), which
drives the same PPO-clipped surrogate GRPO uses. See src/qwen3_rlvr/rl/mopd.py.

Single-dataset note: set `mopd.teacher_paths` to a checkpoint that is genuinely
stronger than the student (e.g. your phase1 GRPO output). With only the base
model as teacher, prefer scripts/train_grpo.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.env import load_project_env
from qwen3_rlvr.logging import setup_logger
from qwen3_rlvr.logging.resource_monitor import ResourceMonitor
from qwen3_rlvr.rl import MOPDConfig, MOPDTrainer

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


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPD on-policy distillation on GSM8K.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--override", action="append", default=[], help="key=value OmegaConf override"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override config output_dir")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument(
        "--teacher-path",
        action="append",
        default=[],
        help="Teacher checkpoint path(s); repeat for multi-teacher. Overrides config.",
    )
    parser.add_argument("--verifier-coef", type=float, default=None, help="Hybrid teacher+verifier")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--monitor-resources",
        action="store_true",
        help="Sample CPU/GPU usage during eval; writes resource_monitor.json to output-dir",
    )
    parser.add_argument(
        "--monitor-interval", type=float, default=2.0, help="Resource sample interval (s)"
    )
    args = parser.parse_args()

    cfg = _apply_overrides(_load_config(args.config), args.override)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    mopd_cfg = _get(cfg, "mopd", default={}) or {}
    dataset_cfg = _get(cfg, "dataset", default={}) or {}
    eval_cfg = _get(cfg, "eval", default={}) or {}
    logging_cfg = _get(cfg, "logging", default={}) or {}
    wandb_cfg = _get(logging_cfg, "wandb", default={}) or {}

    eval_recipes = eval_cfg.get("recipes")
    if not eval_recipes:
        eval_recipe = eval_cfg.get("recipe")
        if eval_recipe:
            eval_recipes = [eval_recipe]
        else:
            dataset = eval_cfg.get("dataset", "gsm8k")
            split = eval_cfg.get("split", "test")
            eval_recipes = [f"{dataset}_{split}"]
    eval_primary_recipe = eval_cfg.get("primary_recipe", eval_recipes[0])

    teacher_paths = args.teacher_path or mopd_cfg.get("teacher_paths", []) or []
    if not teacher_paths:
        parser.error(
            "No teacher specified. Set mopd.teacher_paths in the config or pass "
            "--teacher-path. For a single dataset, use your best GRPO/RLVR checkpoint."
        )

    trainer_cfg = MOPDConfig(
        model_path=cfg["model"],
        output_dir=cfg["output_dir"],
        split=dataset_cfg.get("split", "train"),
        recipe=dataset_cfg.get("recipe"),
        max_samples=dataset_cfg.get("max_samples"),
        max_steps=args.max_steps or mopd_cfg.get("max_steps", 200),
        batch_size=args.batch_size or mopd_cfg.get("batch_size", 2),
        grad_accum_steps=args.grad_accum_steps or mopd_cfg.get("grad_accum_steps", 1),
        n_generations=mopd_cfg.get("n_generations", 8),
        lr=args.lr if args.lr is not None else mopd_cfg.get("lr", 1e-6),
        temperature=mopd_cfg.get("temperature", 0.7),
        max_new_tokens=mopd_cfg.get("max_new_tokens", 512),
        grad_clip=mopd_cfg.get("grad_clip", 1.0),
        dtype=mopd_cfg.get("dtype", "bfloat16"),
        seed=cfg.get("seed", 42),
        eval_batch_size=eval_cfg.get("eval_batch_size", 32),
        eval_every_steps=eval_cfg.get("every_steps", 50),
        eval_recipes=list(eval_recipes),
        eval_primary_recipe=eval_primary_recipe,
        eval_split=eval_cfg.get("split", "test"),
        eval_max_samples=eval_cfg.get("max_samples", 100),
        eval_k=eval_cfg.get("k", [1, 8]),
        eval_n_generations=eval_cfg.get("n_generations", max(eval_cfg.get("k", [1, 8]))),
        eval_max_new_tokens=eval_cfg.get("max_new_tokens", mopd_cfg.get("max_new_tokens", 512)),
        log_every_steps=logging_cfg.get("log_every_steps", 10),
        log_samples_every=logging_cfg.get("log_samples_every", 50),
        sample_table_size=logging_cfg.get("sample_table_size", 8),
        save_every_steps=mopd_cfg.get("save_every_steps", eval_cfg.get("every_steps", 50)),
        wandb_project=None if args.no_wandb else wandb_cfg.get("project"),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_name=args.wandb_name or wandb_cfg.get("name"),
        wandb_tags=(
            [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            if args.wandb_tags
            else wandb_cfg.get("tags")
        ),
        # --- MOPD-specific ---
        teacher_paths=list(teacher_paths),
        teacher_weights=mopd_cfg.get("teacher_weights"),
        clip_eps=mopd_cfg.get("clip_eps", 0.2),
        normalize_advantages=mopd_cfg.get("normalize_advantages", False),
        adv_clip=mopd_cfg.get("adv_clip"),
        verifier_coef=(
            args.verifier_coef
            if args.verifier_coef is not None
            else mopd_cfg.get("verifier_coef", 0.0)
        ),
        mopd_epochs=mopd_cfg.get("mopd_epochs", 1),
    )

    logger.info(f"Training MOPD: student={trainer_cfg.model_path}")
    logger.info(f"Teacher(s): {trainer_cfg.teacher_paths} (weights={trainer_cfg.teacher_weights})")
    logger.info(
        f"verifier_coef={trainer_cfg.verifier_coef} "
        f"normalize_advantages={trainer_cfg.normalize_advantages} adv_clip={trainer_cfg.adv_clip}"
    )
    logger.info(f"Output: {trainer_cfg.output_dir}")
    with ResourceMonitor(
        trainer_cfg.output_dir + "/resource_monitor.json",
        interval_s=args.monitor_interval,
        label="mopd",
    ) as monitor:
        MOPDTrainer(trainer_cfg).train()
    monitor.print_summary()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
