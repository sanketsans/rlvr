"""Curriculum sampling for SFT on rejection-sampled GSM8K data."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from qwen3_rlvr.data.base import SFTExample


@dataclass
class CurriculumConfig:
    enabled: bool = False
    processed_prompt_ids_path: Optional[str] = None
    steps_per_phase: Optional[int] = None
    # original_fraction: float = 0.2 # not used anymore
    phases: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"easy": 0.8, "mid": 0.2, "hard": 0.0, "very_hard": 0.0},
            {"easy": 0.4, "mid": 0.4, "hard": 0.1, "very_hard": 0.1},
            {"easy": 0.1, "mid": 0.5, "hard": 0.25, "very_hard": 0.15},
        ]
    )

    def to_schedule(self) -> CurriculumSchedule:
        return CurriculumSchedule(
            phases=self.phases,
            use_uniform_final=True,
        )


@dataclass(frozen=True)
class ProblemGroup:
    prompt_id: int
    difficulty: str
    example_indices: Tuple[int, ...]


@dataclass
class CurriculumSchedule:
    """Per-epoch difficulty mix for rejection-sampled problems."""

    phases: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"easy": 0.8, "mid": 0.2, "hard": 0.0, "very_hard": 0.0},
            {"easy": 0.4, "mid": 0.4, "hard": 0.1, "very_hard": 0.1},
            {"easy": 0.1, "mid": 0.5, "hard": 0.25, "very_hard": 0.15},
        ]
    )
    # original_fraction: float = 0.2 # not used anymore
    use_uniform_final: bool = True


def build_problem_groups(
    examples: Sequence[SFTExample],
) -> tuple[
    Dict[str, List[ProblemGroup]],
    List[ProblemGroup],
]:
    """
    Group all examples by prompt_id and bucket problems by difficulty.

    Args:
        examples: List of SFTExamples.

    Returns:
        problems_by_difficulty: Dictionary of difficulty to list of ProblemGroups. Each ProblemGroup contains the prompt_id and the indices of the examples for that prompt.
        all_groups: List of all ProblemGroups. This is used for uniform sampling when all phases are over or no probabilities are provided.
    """

    examples_by_prompt: Dict[int, List[int]] = {}  # contains all examples for each prompt
    prompt_difficulty: Dict[int, str] = {}  # difficulty of each prompt

    for idx, ex in enumerate(examples):
        if ex.prompt_id is None:
            continue
        examples_by_prompt.setdefault(ex.prompt_id, []).append(idx)
        # Difficulty is a property of the problem.
        if ex.difficulty is not None:
            prompt_difficulty[ex.prompt_id] = ex.difficulty

    problems_by_difficulty: Dict[str, List[ProblemGroup]] = {
        "easy": [],
        "mid": [],
        "hard": [],
        "very_hard": [],
    }
    all_groups: List[ProblemGroup] = []
    for prompt_id, indices in examples_by_prompt.items():
        difficulty = prompt_difficulty.get(prompt_id, "very_hard")

        if difficulty not in problems_by_difficulty:
            difficulty = "very_hard"

        group = ProblemGroup(
            prompt_id=prompt_id,
            difficulty=difficulty,
            example_indices=tuple(indices),
        )

        problems_by_difficulty[difficulty].append(group)
        all_groups.append(group)

    return problems_by_difficulty, all_groups


def schedule_for_phase(schedule: CurriculumSchedule, phase: int) -> Dict[str, float]:
    if schedule.use_uniform_final and phase >= len(schedule.phases):
        return {}
    if phase < len(schedule.phases):
        return schedule.phases[phase]
    return schedule.phases[-1]


def _normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(probs.values())
    if total <= 0:
        return probs
    return {key: value / total for key, value in probs.items()}


def _sample_difficulty(rng: random.Random, probs: Dict[str, float]) -> str:
    labels = list(probs.keys())
    weights = [probs[label] for label in labels]
    return rng.choices(labels, weights=weights, k=1)[0]


def sample_example_index(
    rng: random.Random,
    *,
    probs: Dict[str, float],
    problems_by_difficulty: Dict[str, List[ProblemGroup]],
    all_groups: List[ProblemGroup],
) -> int:
    """
    Sample one example: pick a problem first, then one accepted solution.

    If no probabilities are provided, sample a random problem from all problems.
    If probabilities are provided, sample a difficulty from the probabilities, then sample a problem from the problems by difficulty.
    Then sample a random example from the problem.
    Did a basic check :
    from collections import Counter
    counter = Counter()

    for step in range(100_000):
        diff = _sample_difficulty(rng, _normalize_probs(probs))
        counter[diff] += 1

    counter is 80% easy and 20% mid based on phase 0 probabilities.

    Args:
        rng: Random number generator.
        probs: Dictionary of difficulty to probability.
        problems_by_difficulty: Dictionary of difficulty to list of ProblemGroups.
        all_groups: List of all ProblemGroups.
    Returns:
        Index of the sampled example.
    """
    if not probs:
        group = rng.choice(all_groups)
        return rng.choice(group.example_indices)

    difficulty = _sample_difficulty(rng, _normalize_probs(probs))
    candidates = problems_by_difficulty.get(difficulty, [])
    if not candidates:
        non_empty = [groups for groups in problems_by_difficulty.values() if groups]
        if not non_empty:
            group = rng.choice(all_groups)
            return rng.choice(group.example_indices)
        candidates = rng.choice(non_empty)

    group = rng.choice(candidates)
    return rng.choice(group.example_indices)


class CurriculumExampleIterator:
    """Infinite iterator with a changing per-epoch difficulty schedule."""

    def __init__(
        self,
        examples: Sequence[SFTExample],
        *,
        schedule: CurriculumSchedule,
        steps_per_phase: int,
        seed: int = 42,
    ):
        self.examples = examples
        self.schedule = schedule
        self.steps_per_phase = max(1, steps_per_phase)
        self.seed = seed
        self.step = 0
        self.problems_by_difficulty, self.all_groups = build_problem_groups(examples)

    @property
    def phase(self) -> int:
        return self.step // self.steps_per_phase

    def __iter__(self) -> Iterator[SFTExample]:
        return self

    def __next__(self) -> SFTExample:
        rng = random.Random(self.seed + self.step)
        probs = schedule_for_phase(self.schedule, self.phase)
        idx = sample_example_index(
            rng,
            probs=probs,
            problems_by_difficulty=self.problems_by_difficulty,
            all_groups=self.all_groups,
        )
        self.step += 1
        return self.examples[idx]


def curriculum_batch_indices(
    examples: Sequence[SFTExample],
    *,
    schedule: CurriculumSchedule,
    steps_per_phase: int,
    batch_size: int,
    step: int,
    seed: int,
    problem_groups: Optional[Tuple[Dict[str, List[ProblemGroup]], List[ProblemGroup]]] = None,
) -> List[int]:
    phase = step // max(1, steps_per_phase)
    probs = schedule_for_phase(schedule, phase)
    if problem_groups is None:
        problem_groups = build_problem_groups(examples)
    problems_by_difficulty, all_groups = problem_groups
    indices: List[int] = []
    for offset in range(batch_size):
        rng = random.Random(seed + step * batch_size + offset)
        indices.append(
            sample_example_index(
                rng,
                probs=probs,
                problems_by_difficulty=problems_by_difficulty,
                all_groups=all_groups,
            )
        )
    return indices
