#!/usr/bin/env python3
"""Curate GSM8K rejection-sampling SFT datasets (top-2 and all-correct variants)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.data.gsm8k import load_gsm8k
from qwen3_rlvr.env import load_project_env
from qwen3_rlvr.generation.rollout import generate_rollouts
from qwen3_rlvr.logging.resource_monitor import ResourceMonitor
from qwen3_rlvr.model.load import load_model_and_tokenizer
from qwen3_rlvr.sft.curation import (
    append_rows_jsonl,
    curate_rollouts,
    load_processed_prompt_ids,
    load_rows_jsonl,
    mark_prompt_processed,
    summarize_rows,
    write_manifest,
)

load_project_env()


def _load_config(path: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _get(cfg: dict, *keys: str, default: Any = None) -> Any:
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate GSM8K rejection-sampling SFT data.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=str, default="/home/coder/Projects/rlvr/outputs/rejection_sampling"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit source prompts for debugging"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip prompts already in progress file"
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    curation_cfg = _get(cfg, "curation", default={}) or {}
    output_dir = Path(
        args.output_dir or cfg.get("output_dir") or ROOT / "outputs" / "rejection_sampling"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg["model"]
    n_rollouts = int(curation_cfg.get("n_rollouts", 16))
    temperature = float(curation_cfg.get("temperature", 0.7))
    max_new_tokens = int(curation_cfg.get("max_new_tokens", 512))
    top_k = int(curation_cfg.get("top_k", 2))
    batch_size = int(curation_cfg.get("batch_size", 4))
    seed = int(cfg.get("seed", 42))
    dtype = curation_cfg.get("dtype", "bfloat16")

    max_samples = (
        args.max_samples if args.max_samples is not None else curation_cfg.get("max_samples")
    )

    top2_jsonl_path = output_dir / "top2.jsonl"
    all_correct_jsonl_path = output_dir / "all_correct.jsonl"
    progress_path = output_dir / "processed_prompt_ids.txt"
    manifest_path = output_dir / "manifest.json"

    if not args.resume:
        for path in (top2_jsonl_path, all_correct_jsonl_path, progress_path):
            if path.exists():
                path.unlink()

    examples = load_gsm8k(split="train", max_samples=max_samples, seed=seed)
    processed = load_processed_prompt_ids(progress_path) if args.resume else set()

    print(f"Model: {model_path}")
    print(f"Source prompts: {len(examples)} (skipping {len(processed)} already processed)")
    print(f"Rollouts per prompt: {n_rollouts}, temperature: {temperature}")

    loaded = load_model_and_tokenizer(model_path=model_path, dtype=dtype, train=False)

    pending: List = [ex for ex in examples if ex.example_id not in processed]
    n_easy_samples, n_medium_samples, n_hard_samples, n_very_hard_samples = 0, 0, 0, 0
    with ResourceMonitor(
        output_dir / "resource_monitor.json", interval_s=10, label="rejection_sampling"
    ) as monitor:
        for start in tqdm(range(0, len(pending), batch_size), desc="Rejection sampling"):
            batch = pending[start : start + batch_size]
            _, completions, _, _, _, _ = generate_rollouts(
                loaded=loaded,
                examples=batch,
                n_generations=n_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed + start,
                tokenize_outputs=False,
            )

            batch_top2 = []
            batch_all = []
            for ex, comp_list in zip(batch, completions):
                top_rows, all_rows, success_ratio = curate_rollouts(
                    prompt_id=ex.example_id,
                    question=ex.question,
                    ground_truth=ex.answer,
                    completions=comp_list,
                    top_k=top_k,
                )
                batch_top2.extend(top_rows)
                batch_all.extend(all_rows)
                if len(all_rows) == 0 and success_ratio == 0.0:
                    n_very_hard_samples += 1
                elif success_ratio >= 0.9:
                    n_easy_samples += 1
                elif success_ratio >= 0.5:
                    n_medium_samples += 1
                else:
                    n_hard_samples += 1
                mark_prompt_processed(progress_path, ex.example_id, success_ratio)

            # batch_top2 and batch_all already contain the success ratio
            append_rows_jsonl(batch_top2, top2_jsonl_path)
            append_rows_jsonl(batch_all, all_correct_jsonl_path)

    monitor.print_summary()

    top2_rows = load_rows_jsonl(top2_jsonl_path)
    top2_count = len(top2_rows)
    all_correct_rows = load_rows_jsonl(all_correct_jsonl_path)
    all_correct_count = len(all_correct_rows)
    top2_summary = summarize_rows(top2_rows)
    all_correct_summary = summarize_rows(all_correct_rows)

    _manifest = write_manifest(
        manifest_path,
        model_path=model_path,
        output_dir=output_dir,
        num_source_prompts=len(examples),
        n_rollouts_per_prompt=n_rollouts,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top2_jsonl_path=top2_jsonl_path,
        all_correct_jsonl_path=all_correct_jsonl_path,
        top2_summary=top2_summary,
        all_correct_summary=all_correct_summary,
        config_path=args.config,
        extra={
            "top2_rows_written": top2_count,
            "all_correct_rows_written": all_correct_count,
            "n_easy_samples": n_easy_samples,
            "n_medium_samples": n_medium_samples,
            "n_hard_samples": n_hard_samples,
            "n_very_hard_samples": n_very_hard_samples,
        },
    )

    print(f"Manifest: {manifest_path}")
    print(f"Unique prompts with >=1 correct (top2): {top2_summary['num_unique_prompts']}")
    print(f"Unique prompts with >=1 correct (all): {all_correct_summary['num_unique_prompts']}")
    print(f"Easy samples: {n_easy_samples} ({n_easy_samples / len(examples) * 100:.2f}%)")
    print(f"Medium samples: {n_medium_samples} ({n_medium_samples / len(examples) * 100:.2f}%)")
    print(f"Hard samples: {n_hard_samples} ({n_hard_samples / len(examples) * 100:.2f}%)")
    print(
        f"Very hard samples: {n_very_hard_samples} ({n_very_hard_samples / len(examples) * 100:.2f}%)"
    )
    print(
        f"Total samples: {n_easy_samples + n_medium_samples + n_hard_samples + n_very_hard_samples} ({n_easy_samples + n_medium_samples + n_hard_samples + n_very_hard_samples / len(examples) * 100:.2f}%)"
    )

    print("Rejection sampling data curation complete.")


if __name__ == "__main__":
    main()
