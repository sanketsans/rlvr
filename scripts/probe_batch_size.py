#!/usr/bin/env python3
"""Probe max generation batch size for a model on the current GPU."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.data.gsm8k import load_gsm8k
from qwen3_rlvr.eval.pass_at_k import _format_prompt
from qwen3_rlvr.logging.resource_monitor import sample_resources
from qwen3_rlvr.model.load import load_model_and_tokenizer


def _try_batch(loaded, prompts: list[str], max_new_tokens: int) -> tuple[bool, float]:
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = loaded.device

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    torch.cuda.reset_peak_memory_stats(device)

    try:
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        return True, peak_gb
    except torch.cuda.OutOfMemoryError:
        return False, 0.0
    finally:
        del inputs
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Find largest generation batch size that fits on GPU.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — batch probing requires a GPU.")

    examples = load_gsm8k(split="test", max_samples=args.max_batch, seed=0)
    loaded = load_model_and_tokenizer(args.model, dtype=args.dtype)

    print(f"Model: {args.model}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"{'batch':>6}  {'ok':>4}  {'peak_gb':>8}")
    print("-" * 24)

    last_ok = 0
    last_peak = 0.0
    batch = args.start_batch

    while batch <= args.max_batch:
        prompts = [
            _format_prompt(loaded.tokenizer, ex.messages)
            for ex in examples[:batch]
        ]
        ok, peak_gb = _try_batch(loaded, prompts, args.max_new_tokens)
        print(f"{batch:6d}  {'yes' if ok else 'no':>4}  {peak_gb:8.2f}")

        if ok:
            last_ok = batch
            last_peak = peak_gb
            batch *= 2
        else:
            break

    snap = sample_resources()
    print("-" * 24)
    print(f"Largest OK batch size: {last_ok} (peak torch alloc {last_peak:.2f} GB)")
    if snap.gpu_mem_total_gb is not None:
        headroom = snap.gpu_mem_total_gb - last_peak
        print(f"GPU total (smi): {snap.gpu_mem_total_gb:.2f} GB, approx headroom at max batch: {headroom:.2f} GB")


if __name__ == "__main__":
    main()
