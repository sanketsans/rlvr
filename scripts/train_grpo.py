#!/usr/bin/env python3
"""Phase 1: GRPO training on GSM8K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.env import load_project_env
from qwen3_rlvr.rl.trainer import GRPOTrainer, TrainerConfig
from qwen3_rlvr.logging.resource_monitor import ResourceMonitor

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
    parser = argparse.ArgumentParser(description="GRPO training on GSM8K.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", action="append", default=[], help="key=value OmegaConf override")
    parser.add_argument("--output-dir", type=str, default=None, help="Override config output_dir")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--monitor-resources",
        action="store_true",
        help="Sample CPU/GPU usage during eval; writes resource_monitor.json to output-dir",
    )
    parser.add_argument("--monitor-interval", type=float, default=2.0, help="Resource sample interval (s)")
    args = parser.parse_args()

    cfg = _apply_overrides(_load_config(args.config), args.override)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    grpo_cfg = _get(cfg, "grpo", default={}) or {}
    dataset_cfg = _get(cfg, "dataset", default={}) or {}
    eval_cfg = _get(cfg, "eval", default={}) or {}
    logging_cfg = _get(cfg, "logging", default={}) or {}
    wandb_cfg = _get(logging_cfg, "wandb", default={}) or {}

    trainer_cfg = TrainerConfig(
        model_path=cfg["model"],
        output_dir=cfg["output_dir"],
        split=dataset_cfg.get("split", "train"),
        max_samples=dataset_cfg.get("max_samples"),
        max_steps=args.max_steps or grpo_cfg.get("max_steps", 200),
        batch_size=args.batch_size or grpo_cfg.get("batch_size", 2),
        grad_accum_steps=args.grad_accum_steps or grpo_cfg.get("grad_accum_steps", 1),
        n_generations=grpo_cfg.get("n_generations", 4),
        lr=args.lr if args.lr is not None else grpo_cfg.get("lr", 1e-6),
        kl_coef=grpo_cfg.get("kl_coef", 0.04),
        temperature=grpo_cfg.get("temperature", 0.7),
        max_new_tokens=grpo_cfg.get("max_new_tokens", 256),
        grad_clip=grpo_cfg.get("grad_clip", 1.0),
        dtype=grpo_cfg.get("dtype", "bfloat16"),
        reinforce=grpo_cfg.get("reinforce", False),
        seed=cfg.get("seed", 42),
        eval_every_steps=eval_cfg.get("every_steps", 50),
        eval_split=eval_cfg.get("split", "test"),
        eval_max_samples=eval_cfg.get("max_samples", 100),
        eval_k=eval_cfg.get("k", [1, 8]),
        eval_n_generations=eval_cfg.get("n_generations", max(eval_cfg.get("k", [1, 8]))),
        eval_max_new_tokens=eval_cfg.get("max_new_tokens", grpo_cfg.get("max_new_tokens", 256)),
        log_every_steps=logging_cfg.get("log_every_steps", 10),
        log_samples_every=logging_cfg.get("log_samples_every", 50),
        sample_table_size=logging_cfg.get("sample_table_size", 8),
        save_every_steps=grpo_cfg.get("save_every_steps", eval_cfg.get("every_steps", 50)),
        wandb_project=None if args.no_wandb else wandb_cfg.get("project"),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_name=args.wandb_name or wandb_cfg.get("name"),
        wandb_tags=(
            [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            if args.wandb_tags
            else wandb_cfg.get("tags")
        ),
    )

    print(f"Training GRPO: {trainer_cfg.model_path}")
    print(f"Output: {trainer_cfg.output_dir}")
    # if args.monitor_resources:
    with ResourceMonitor(trainer_cfg.output_dir + "/resource_monitor.json", interval_s=args.monitor_interval, label="grpo") as monitor:
        GRPOTrainer(trainer_cfg).train()
    monitor.print_summary()
    # else:
    #     GRPOTrainer(trainer_cfg).train()
    print("Training complete.")


if __name__ == "__main__":
    main()
