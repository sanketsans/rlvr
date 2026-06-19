"""GSM8K dataset loading and prompt formatting."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from datasets import load_dataset

from qwen3_rlvr.data.base import SFTExample, VerifiableExample
from qwen3_rlvr.rewards.extract import extract_reference_answer


def load_gsm8k(
    split: str = "test",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[VerifiableExample]:
    """Load GSM8K from HuggingFace Hub."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    return [
        VerifiableExample(
            example_id=i,
            question=row["question"],
            answer=extract_reference_answer(row["answer"], source="gsm8k"),
            source="gsm8k",
        )
        for i, row in enumerate(dataset)
    ]


def load_gsm8k_sft(
    jsonl_path: str | Path,
    max_samples: Optional[int] = None,
    seed: int = 42,
    variant: Optional[str] = None,
    include_original: bool = True,
    original_split: str = "train",
    original_max_samples: Optional[int] = None,
    processed_prompt_ids_path: str | Path | None = None,
) -> List[SFTExample]:
    """Load curated GSM8K rejection-sampling JSONL rows for SFT."""
    from qwen3_rlvr.sft.dataset import load_gsm8k_sft as _load

    return _load(
        jsonl_path=jsonl_path,
        max_samples=max_samples,
        seed=seed,
        variant=variant,
        include_original=include_original,
        original_split=original_split,
        original_max_samples=original_max_samples,
        processed_prompt_ids_path=processed_prompt_ids_path,
    )
