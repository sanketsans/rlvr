"""AIME dataset loading."""

from __future__ import annotations

from typing import List, Optional

from datasets import concatenate_datasets, load_dataset

from qwen3_rlvr.data.base import VerifiableExample
from qwen3_rlvr.rewards.extract import extract_reference_answer


def _normalize_aime_row(row: dict) -> tuple[str, str]:
    question = row.get("problem") or row.get("Problem") or row.get("question", "")
    raw_answer = row.get("answer") or row.get("Answer") or row.get("solution", "")
    return question, extract_reference_answer(str(raw_answer), source="aime")


def load_aime(
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
    hub_id: str = "HuggingFaceH4/aime_2024",
) -> List[VerifiableExample]:
    # This only has 30 samples and is anyway called "train"set.
    # But using this as the test for our AIME evals. 

    dataset = load_dataset(hub_id, split=split)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    return [
        VerifiableExample(
            example_id=i,
            question=question,
            answer=answer,
            source="aime",
        )
        for i, (question, answer) in enumerate(_normalize_aime_row(row) for row in dataset)
    ]


def load_aime_combined(
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[VerifiableExample]:
    """Load and dedupe AIME problems from common public HF sources."""
    parts = [
        load_dataset("AI-MO/aimo-validation-aime", split=split),
        # load_dataset("Maxwell-Jia/AIME_2024", split=split),
        # load_dataset("HuggingFaceH4/aime_2024", split=split),
    ]
    dataset = concatenate_datasets(parts)

    seen_questions: set[str] = set()
    examples: List[VerifiableExample] = []
    for row in dataset:
        question, answer = _normalize_aime_row(row)
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        examples.append(
            VerifiableExample(
                example_id=len(examples),
                question=question,
                answer=answer,
                source="aime",
            )
        )

    if max_samples is not None and max_samples < len(examples):
        import random

        rng = random.Random(seed)
        examples = rng.sample(examples, k=max_samples)
        examples = [
            VerifiableExample(example_id=i, question=ex.question, answer=ex.answer, source=ex.source)
            for i, ex in enumerate(examples)
        ]

    return examples
