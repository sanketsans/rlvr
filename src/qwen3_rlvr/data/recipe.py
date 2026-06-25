"""Dataset recipes for mixed RLVR training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from qwen3_rlvr.data.aime import load_aime, load_aime_combined
from qwen3_rlvr.data.base import VerifiableExample
from qwen3_rlvr.data.gsm8k import load_gsm8k
from qwen3_rlvr.data.math import load_math

DatasetLoader = Callable[..., List[VerifiableExample]]


@dataclass(frozen=True)
class RecipeSource:
    name: str
    split: str
    weight: float = 1.0
    max_samples: Optional[int] = None


LOADERS: Dict[str, DatasetLoader] = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "aime": load_aime,
    "aime_combined": load_aime_combined,
}

RECIPES: Dict[str, List[RecipeSource]] = {
    "gsm8k_train": [RecipeSource("gsm8k", "train")],
    "gsm8k_test": [RecipeSource("gsm8k", "test")],
    "math_train": [RecipeSource("math", "train")],
    "math_test": [RecipeSource("math", "test")],
    "aime_test": [RecipeSource("aime", "train")],
    "aime_combined_train": [RecipeSource("aime_combined", "train")],
    "gsm8k_math_train": [
        RecipeSource("gsm8k", "train", weight=1.0),
        RecipeSource("math", "train", weight=1.0),
    ],
    "gsm8k_math_aime_train": [
        RecipeSource("gsm8k", "train", weight=1.0),
        RecipeSource("math", "train", weight=1.0),
        RecipeSource("aime_combined", "train", weight=0.25),
    ],
}


def list_recipes() -> List[str]:
    return sorted(RECIPES)


def _subsample(
    examples: List[VerifiableExample], max_samples: int, seed: int
) -> List[VerifiableExample]:
    if max_samples >= len(examples):
        return examples
    rng = random.Random(seed)
    picked = rng.sample(examples, k=max_samples)
    return [
        VerifiableExample(example_id=i, question=ex.question, answer=ex.answer, source=ex.source)
        for i, ex in enumerate(picked)
    ]


def _allocate_counts(total: int, weights: List[float]) -> List[int]:
    if total <= 0:
        return [0 for _ in weights]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Recipe source weights must sum to a positive value.")

    raw = [total * (weight / weight_sum) for weight in weights]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    if remainder:
        ranked = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
        for idx in ranked[:remainder]:
            counts[idx] += 1
    return counts


def load_recipe(
    recipe: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[VerifiableExample]:
    if recipe not in RECIPES:
        known = ", ".join(list_recipes())
        raise ValueError(f"Unknown recipe '{recipe}'. Available recipes: {known}")

    sources = RECIPES[recipe]
    loaded_groups: List[List[VerifiableExample]] = []
    weights = [source.weight for source in sources]

    for source in sources:
        if source.name not in LOADERS:
            raise ValueError(f"Unknown dataset loader '{source.name}' in recipe '{recipe}'.")
        loader = LOADERS[source.name]
        per_source_cap = source.max_samples
        loaded_groups.append(loader(split=source.split, max_samples=per_source_cap, seed=seed))

    if max_samples is None:
        merged: List[VerifiableExample] = []
        for group in loaded_groups:
            merged.extend(group)
        rng = random.Random(seed)
        rng.shuffle(merged)
        return [
            VerifiableExample(
                example_id=i, question=ex.question, answer=ex.answer, source=ex.source
            )
            for i, ex in enumerate(merged)
        ]

    counts = _allocate_counts(max_samples, weights)
    merged = []
    for group, count in zip(loaded_groups, counts):
        merged.extend(_subsample(group, count, seed=seed))
    rng = random.Random(seed)
    rng.shuffle(merged)
    return [
        VerifiableExample(example_id=i, question=ex.question, answer=ex.answer, source=ex.source)
        for i, ex in enumerate(merged)
    ]


def load_dataset_by_name(
    name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[VerifiableExample]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {sorted(LOADERS)}")
    return LOADERS[name](split=split, max_samples=max_samples, seed=seed)
