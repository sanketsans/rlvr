"""Shared types and prompts for RLVR training examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

SYSTEM_PROMPT = {
    "gsm8k": "You are a helpful assistant. Solve the math problem step by step. End your response with the final answer on its own line in the format: #### <answer>",
    "math": "You are a helpful assistant. Solve the math problem step by step, and put your final answer within '\\boxed{}'. <answer>",
    "aime": "You are a helpful assistant. Solve the math problem step by step, and put your final answer within '\\boxed{}'. <answer>",
}


@dataclass(frozen=True)
class VerifiableExample:
    example_id: int
    question: str
    answer: str
    source: str

    @property
    def messages(self) -> List[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT[self.source]},
            {"role": "user", "content": self.question},
        ]


@dataclass(frozen=True)
class SFTExample:
    example_id: int
    question: str
    completion: str
    source: str
    prompt_id: int | None = None
    difficulty: str | None = None
    data_source: str = "rejection"

    @property
    def messages(self) -> List[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT[self.source]},
            {"role": "user", "content": self.question},
            {"role": "assistant", "content": self.completion},
        ]
