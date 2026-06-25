"""MATH (hendrycks_math) dataset loading."""

from __future__ import annotations

from typing import List, Optional

from datasets import concatenate_datasets, load_dataset

from qwen3_rlvr.data.base import VerifiableExample
from qwen3_rlvr.rewards.extract import extract_reference_answer

MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def load_math(
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[VerifiableExample]:
    parts = [
        load_dataset("EleutherAI/hendrycks_math", subject, split=split) for subject in MATH_SUBJECTS
    ]
    dataset = concatenate_datasets(parts)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    return [
        VerifiableExample(
            example_id=i,
            question=row["problem"],
            answer=extract_reference_answer(row["solution"], source="math"),
            source="math",
        )
        for i, row in enumerate(dataset)
    ]
