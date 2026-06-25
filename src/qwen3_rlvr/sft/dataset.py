"""SFT dataset loading from curated JSONL files."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from qwen3_rlvr.data.base import SFTExample
from qwen3_rlvr.generation.prompts import format_prompt
from qwen3_rlvr.logging.logger import setup_logger
from qwen3_rlvr.sft.curation import (
    CuratedRow,
    classify_difficulty,
    load_prompt_success_ratios,
    load_rows_jsonl,
)

logger = setup_logger(__name__)


class SFTTokenDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        # same idea as used in the GRPO objectiv e- we do not want to tokenize prompt + completion together, since BPE tokenization works with neighboring tokens.
        prompt_messages = ex.messages[:2]
        prompt_text = format_prompt(self.tokenizer, prompt_messages)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        completion_text = ex.completion
        if self.tokenizer.eos_token:
            completion_text += self.tokenizer.eos_token

        completion_ids = self.tokenizer(
            completion_text,
            add_special_tokens=False,
        )["input_ids"]

        # Preserve completion as much as possible; truncate prompt from the left.
        if len(completion_ids) >= self.max_seq_length:
            input_ids = completion_ids[: self.max_seq_length]
            labels = input_ids.copy()
        else:
            max_prompt_len = self.max_seq_length - len(completion_ids)
            prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "length": len(input_ids),
        }


def _collate(batch: List[dict], pad_token_id: int) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        input_ids.append(
            torch.cat(
                [
                    torch.full(
                        (pad_len,), pad_token_id, dtype=torch.long
                    ),  # replicate the pad token to the left of the input_ids
                    item["input_ids"],  # these are the input_ids over prompt + completion tokens.
                ]
            )
        )
        attention_mask.append(
            torch.cat(
                [
                    torch.zeros(
                        pad_len, dtype=torch.long
                    ),  # replicate the pad token to the left of the attention_mask
                    item[
                        "attention_mask"
                    ],  # these are the attention_mask over prompt + completion tokens.
                ]
            )
        )
        labels.append(
            torch.cat(
                [
                    torch.full((pad_len,), -100, dtype=torch.long),
                    item["labels"],  # these are the labels over prompt + completion tokens.
                ]
            )
        )
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def load_gsm8k_original_sft(
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
    prompt_ratios: Optional[Dict[int, float]] = {},
) -> List[SFTExample]:
    """Load official GSM8K chain-of-thought answers for SFT."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    logger.info(f"Loaded {len(dataset)} original examples from {split}")
    return [
        SFTExample(
            example_id=i,
            question=row["question"],
            completion=row["answer"],
            source="gsm8k",
            prompt_id=i,
            difficulty=classify_difficulty(prompt_ratios.get(i, 0.0)),
            data_source="original",
        )
        for i, row in enumerate(dataset)
    ]


def _curated_row_to_example(
    row: CuratedRow,
    example_id: int,
    prompt_ratios: Dict[int, float],
) -> SFTExample:
    success_ratio = row.success_ratio
    if success_ratio is None and row.prompt_id in prompt_ratios:
        success_ratio = prompt_ratios[row.prompt_id]
    difficulty = classify_difficulty(success_ratio if success_ratio is not None else 0.0)
    return SFTExample(
        example_id=example_id,
        question=row.question,
        completion=row.completion,
        source="gsm8k",
        prompt_id=row.prompt_id,
        difficulty=difficulty,
        data_source="rejection",
    )


def load_gsm8k_sft(
    jsonl_path: str | Path,
    *,
    max_samples: Optional[int] = None,
    seed: int = 42,
    variant: Optional[str] = None,
    include_original: bool = True,
    original_split: str = "train",
    original_max_samples: Optional[int] = None,
    processed_prompt_ids_path: str | Path | None = None,
) -> List[SFTExample]:
    """Load curated rejection-sampling JSONL and append original GSM8K SFT rows."""
    path = Path(jsonl_path)

    prompt_ratios: Dict[int, float] = {}
    if path.is_file():
        if processed_prompt_ids_path is not None:
            prompt_ratios = load_prompt_success_ratios(Path(processed_prompt_ids_path))

        rows = load_rows_jsonl(path)
        if variant is not None:
            rows = [row for row in rows if row.variant == variant]

        examples = [
            _curated_row_to_example(row, example_id=i, prompt_ratios=prompt_ratios)
            for i, row in enumerate(rows)
        ]

        logger.info(f"Loaded {len(examples)} examples from {path}")
    else:
        examples = []
        logger.info(f"No JSONL file found at {path}.")

    if include_original:
        original = load_gsm8k_original_sft(
            split=original_split,
            max_samples=original_max_samples,
            seed=seed,
            prompt_ratios=prompt_ratios,
        )
        offset = len(examples)
        examples.extend(
            SFTExample(
                example_id=offset + i,
                question=ex.question,
                completion=ex.completion,
                source=ex.source,
                prompt_id=ex.prompt_id,
                difficulty=ex.difficulty,
                data_source=ex.data_source,
            )
            for i, ex in enumerate(original)
        )
    logger.info(f"Total examples: {len(examples)}")

    if max_samples is not None and max_samples < len(examples):
        rng = random.Random(seed)
        picked = rng.sample(range(len(examples)), k=max_samples)
        logger.info(f"Picked {len(picked)} examples from {len(examples)}")
        return [examples[i] for i in sorted(picked)]

    return examples


def _resolve_variant_jsonl_path(
    jsonl_path: str | Path,
    *,
    variant: str,
    manifest: dict,
) -> Path:
    path = Path(jsonl_path)
    if path.is_file():
        return path

    output_dir = Path(manifest.get("output_dir", path.parent))
    candidates = [
        output_dir / f"{variant}.jsonl",
        output_dir / "top2.jsonl" if variant == "top2" else output_dir / "all_correct.jsonl",
        path.with_suffix(".jsonl"),
        path.parent / f"{variant}.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            logger.info(f"Found SFT JSONL for variant '{variant}': {candidate}")
            return candidate
    raise FileNotFoundError(
        f"SFT JSONL not found for variant '{variant}'. Tried: {[str(c) for c in candidates]}"
    )


def load_sft_from_manifest(
    manifest_path: str | Path,
    variant: str = "top2",
    max_samples: Optional[int] = None,
    seed: int = 42,
    include_original: bool = True,
    original_split: str = "train",
    original_max_samples: Optional[int] = None,
    processed_prompt_ids_path: str | Path | None = None,
) -> List[SFTExample]:
    from qwen3_rlvr.sft.curation import load_manifest

    manifest = load_manifest(manifest_path)
    if variant not in manifest["variants"]:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {list(manifest['variants'].keys())}"
        )

    jsonl_path = _resolve_variant_jsonl_path(
        manifest["variants"][variant]["path"],
        variant=variant,
        manifest=manifest,
    )
    if processed_prompt_ids_path is None:
        output_dir = Path(manifest.get("output_dir", Path(manifest_path).parent))
        candidate = output_dir / "processed_prompt_ids.txt"
        if candidate.is_file():
            processed_prompt_ids_path = candidate
        else:
            raise FileNotFoundError(f"processed_prompt_ids.txt not found in {output_dir}")

    return load_gsm8k_sft(
        jsonl_path=jsonl_path,
        max_samples=max_samples,
        seed=seed,
        variant=variant,
        include_original=include_original,
        original_split=original_split,
        original_max_samples=original_max_samples,
        processed_prompt_ids_path=processed_prompt_ids_path,
    )
