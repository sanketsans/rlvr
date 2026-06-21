"""Recipe-based evaluation for RLVR verifiable tasks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm
from qwen3_rlvr.data.recipe import load_recipe
from qwen3_rlvr.eval.pass_at_k import compute_pass_at_k
from qwen3_rlvr.generation.rollout import generate_rollouts
from qwen3_rlvr.model.load import LoadedModel
from qwen3_rlvr.rewards.extract import answers_match


def _aggregate_metrics(
    per_example_masks: List[List[bool]],
    k_values: Sequence[int],
    method: str,
) -> Dict[str, float]:
    aggregate = {f"pass@{k}": 0.0 for k in k_values}
    num_correct_total = 0.0

    for mask in per_example_masks:
        q_metrics = compute_pass_at_k(mask, k_values, method=method)
        for key, val in q_metrics.items():
            aggregate[key] += val
        num_correct_total += sum(mask)

    n = max(len(per_example_masks), 1)
    for key in aggregate:
        aggregate[key] /= n
    aggregate["accuracy"] = aggregate.get("pass@1", 0.0)
    aggregate["avg_num_correct"] = num_correct_total / n
    return aggregate


def evaluate_recipe_quick(
    loaded: LoadedModel,
    recipe: str,
    max_samples: Optional[int] = None,
    n_generations: int = 8,
    k_values: Optional[Sequence[int]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42,
    question_batch_size: int = 4,
    method: str = "unbiased",
) -> Dict[str, Any]:
    """Fast pass@k evaluation on a recipe using an already-loaded model."""
    if k_values is None:
        k_values = [1, 8]
    if n_generations < max(k_values):
        raise ValueError(f"n_generations={n_generations} must be >= max(k)={max(k_values)}")

    examples = load_recipe(recipe, max_samples=max_samples, seed=seed)
    per_example_masks: List[List[bool]] = []
    source_masks: Dict[str, List[List[bool]]] = defaultdict(list)

    was_training = loaded.model.training
    loaded.model.eval()

    try:
        for start in tqdm(range(0, len(examples), question_batch_size), desc="Generating rollouts"):
            batch = examples[start : start + question_batch_size]
            _, completions, _, _, _, _ = generate_rollouts(
                loaded=loaded,
                examples=batch,
                n_generations=n_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed + start,
                tokenize_outputs=False,
            )
            for ex, comp_list in zip(batch, completions):
                mask = [answers_match(c, ex.answer) for c in comp_list]
                per_example_masks.append(mask)
                source_masks[ex.source].append(mask)
    finally:
        if was_training:
            loaded.model.train()

    metrics = _aggregate_metrics(per_example_masks, k_values, method=method)
    by_source = {
        source: {
            **_aggregate_metrics(masks, k_values, method=method),
            "num_examples": len(masks),
        }
        for source, masks in sorted(source_masks.items())
    }

    return {
        "recipe": recipe,
        "num_examples": len(examples),
        "n_generations": n_generations,
        "temperature": temperature,
        "k_values": list(k_values),
        "method": method,
        **metrics,
        "by_source": by_source,
    }
