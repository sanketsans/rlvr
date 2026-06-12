"""GSM8K dataset loading and prompt formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the math problem step by step. "
    "End your response with the final numeric answer on its own line in the format: #### <answer>"
)


@dataclass(frozen=True)
class Gsm8kExample:
    example_id: int
    question: str
    answer: str

    @property
    def messages(self) -> List[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.question},
        ]


def load_gsm8k(
    split: str = "test",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Gsm8kExample]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    return [
        Gsm8kExample(example_id=i, question=row["question"], answer=row["answer"])
        for i, row in enumerate(dataset)
    ]
