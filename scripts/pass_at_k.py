#!/usr/bin/env python3
"""Phase 0: Pass@K evaluation on GSM8K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.eval.pass_at_k import evaluate_pass_at_k, save_results
from qwen3_rlvr.logging.resource_monitor import ResourceMonitor
from qwen3_rlvr.logging.wandb_logger import log_pass_at_k_to_wandb


def _parse_k_values(raw: Optional[str], cfg_values: Optional[List[int]]) -> List[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    if cfg_values:
        return [int(x) for x in cfg_values]
    return [1, 8, 16]


def _load_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pass@K on GSM8K.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--model", type=str, default=None, help="HF model path or hub id")
    parser.add_argument("--split", type=str, default=None, choices=["train", "test"])
    parser.add_argument("--k", type=str, default=None, help="Comma-separated k values, e.g. 1,8,16")
    parser.add_argument("--n-generations", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--question-batch-size",
        type=int,
        default=None,
        help="Number of GSM8K prompts per generate() call (tune with probe_batch_size.py)",
    )
    parser.add_argument(
        "--dtype", type=str, default=None, choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--method", type=str, default=None, choices=["unbiased", "first_k"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb-expt-name", type=str, default=None, help="W&B run name override")
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

    cfg = _load_config(args.config)

    model_path = args.model or cfg.get("model")
    if not model_path:
        raise SystemExit("--model is required (or set `model` in --config)")

    passk_cfg = cfg.get("passk", {})
    dataset_cfg = cfg.get("dataset", {})
    logging_cfg = cfg.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {})

    k_values = _parse_k_values(args.k, passk_cfg.get("k"))
    n_generations = args.n_generations or passk_cfg.get("n_generations", max(k_values))
    if n_generations < max(k_values):
        raise SystemExit(f"--n-generations must be >= max(k)={max(k_values)}")

    output_dir = (
        args.output_dir or cfg.get("output_dir") or str(ROOT / "outputs" / "pass_at_k_default")
    )
    monitor_path = Path(output_dir) / "resource_monitor.json"

    def _run_eval():
        return evaluate_pass_at_k(
            model_path=model_path,
            split=args.split or dataset_cfg.get("split", "test"),
            max_samples=args.max_samples
            if args.max_samples is not None
            else dataset_cfg.get("max_samples"),
            k_values=k_values,
            n_generations=n_generations,
            temperature=args.temperature
            if args.temperature is not None
            else passk_cfg.get("temperature", 0.7),
            max_new_tokens=args.max_new_tokens or passk_cfg.get("max_new_tokens", 512),
            dtype=args.dtype or passk_cfg.get("dtype", "bfloat16"),
            seed=args.seed if args.seed is not None else cfg.get("seed", 42),
            method=args.method or passk_cfg.get("method", "unbiased"),
            question_batch_size=args.question_batch_size or passk_cfg.get("question_batch_size", 4),
        )

    if args.monitor_resources:
        with ResourceMonitor(
            monitor_path, interval_s=args.monitor_interval, label="pass_at_k"
        ) as monitor:
            result = _run_eval()
        monitor.print_summary()
    else:
        result = _run_eval()

    summary_path = save_results(result, output_dir)

    print(f"Model: {result.model_path}")
    print(f"Examples: {result.num_examples}")
    print(f"Generations per question: {result.n_generations}")
    for key, value in sorted(result.pass_at_k.items()):
        print(f"{key}: {value:.4f}")
    print(f"avg_num_correct: {result.avg_num_correct:.4f}")
    print(f"Saved summary: {summary_path}")

    use_wandb = not args.no_wandb and wandb_cfg.get("project")
    if use_wandb:
        log_pass_at_k_to_wandb(
            result=result,
            project=wandb_cfg["project"],
            name=args.wandb_expt_name
            or wandb_cfg.get("name", "pass_at_k") + f"_max_samples_{args.max_samples}",
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags"),
        )
        print("Logged metrics to W&B")


if __name__ == "__main__":
    main()
