"""In-loop GSM8K evaluation using an already-loaded model."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from qwen3_rlvr.eval.pass_at_k import compute_pass_at_k
from qwen3_rlvr.eval.recipe_eval import evaluate_recipe_quick
from qwen3_rlvr.model.load import LoadedModel

__all__ = ["compute_pass_at_k", "evaluate_gsm8k_quick", "evaluate_recipe_quick"]


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
    """Backward-compatible wrapper around recipe eval for GSM8K test."""
    del split  # recipe encodes split
    result = evaluate_recipe_quick(
        loaded=loaded,
        recipe="gsm8k_test",
        max_samples=max_samples,
        n_generations=n_generations,
        k_values=k_values,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        question_batch_size=question_batch_size,
    )
    return {k: v for k, v in result.items() if k not in {"recipe", "by_source", "k_values", "method"}}
