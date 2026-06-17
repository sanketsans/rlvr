"""In-loop GSM8K evaluation using an already-loaded model."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from qwen3_rlvr.data.gsm8k import load_gsm8k
from qwen3_rlvr.eval.pass_at_k import compute_pass_at_k
from qwen3_rlvr.generation.rollout import generate_rollouts
from qwen3_rlvr.model.load import LoadedModel
from qwen3_rlvr.rewards.extract import answers_match


def evaluate_gsm8k_quick(
    loaded: LoadedModel,
    split: str = "test",
    max_samples: int = 100,
    n_generations: int = 8,
    k_values: Optional[Sequence[int]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42,
    question_batch_size: int = 4,
) -> Dict[str, float]:
    """Fast GSM8K pass@k on a loaded policy without reloading weights."""
    if k_values is None:
        k_values = [1, 8]
    if n_generations < max(k_values):
        raise ValueError(f"n_generations={n_generations} must be >= max(k)={max(k_values)}")

    examples = load_gsm8k(split=split, max_samples=max_samples, seed=seed)
    aggregate = {f"pass@{k}": 0.0 for k in k_values}
    num_correct_total = 0.0

    was_training = loaded.model.training
    loaded.model.eval()

    try:
        for start in range(0, len(examples), question_batch_size):
            batch = examples[start : start + question_batch_size]
            _, completions, _, _, _ = generate_rollouts(
                loaded=loaded,
                examples=batch,
                n_generations=n_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed + start,
            )
            for ex, comp_list in zip(batch, completions):
                mask = [answers_match(c, ex.answer) for c in comp_list]
                q_metrics = compute_pass_at_k(mask, k_values, method="unbiased")
                for key, val in q_metrics.items():
                    aggregate[key] += val
                num_correct_total += sum(mask)
    finally:
        if was_training:
            loaded.model.train()

    n = max(len(examples), 1)
    for key in aggregate:
        aggregate[key] /= n
    aggregate["accuracy"] = aggregate.get("pass@1", 0.0)
    aggregate["avg_num_correct"] = num_correct_total / n
    return aggregate
