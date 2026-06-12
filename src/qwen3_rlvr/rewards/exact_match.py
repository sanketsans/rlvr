"""Binary exact-match rewards for verifiable tasks."""

from __future__ import annotations

from typing import List, Sequence

from qwen3_rlvr.rewards.extract import answers_match


def exact_match_rewards(completions: Sequence[str], reference: str) -> List[float]:
    return [1.0 if answers_match(c, reference) else 0.0 for c in completions]
