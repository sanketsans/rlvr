"""Pass@K evaluation for verifiable math tasks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from itertools import islice

from qwen3_rlvr.data.base import VerifiableExample
from qwen3_rlvr.data.recipe import load_dataset_by_name, load_recipe
from qwen3_rlvr.generation.prompts import format_prompt
from qwen3_rlvr.generation.rollout import generate_rollouts
from qwen3_rlvr.model.load import LoadedModel, load_model_and_tokenizer
from qwen3_rlvr.rewards.extract import answers_match


@dataclass
class QuestionResult:
    example_id: int
    question: str
    ground_truth: str
    completions: List[str]
    correct_mask: List[bool]
    num_correct: int


@dataclass
class PassAtKResult:
    model_path: str
    dataset: str
    split: str
    n_generations: int
    temperature: float
    k_values: List[int]
    num_examples: int
    pass_at_k: Dict[str, float]
    avg_num_correct: float
    per_question: List[QuestionResult] = field(repr=False)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_summary_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "dataset": self.dataset,
            "split": self.split,
            "n_generations": self.n_generations,
            "temperature": self.temperature,
            "k_values": self.k_values,
            "num_examples": self.num_examples,
            "pass_at_k": self.pass_at_k,
            "avg_num_correct": self.avg_num_correct,
            "created_at": self.created_at,
        }

def batched(iterable, batch_size):
    iterator = iter(iterable)

    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def compute_pass_at_k(
    correctness_per_generation: Sequence[bool],
    k_values: Sequence[int],
    method: str = "unbiased"
) -> Dict[str, float]:
    """
    Compute pass@k metric(s) for a single problem, given a set of generations.
    
    Args:
        correctness_per_generation: Sequence of booleans, where each value indicates
            whether a single generation (sample) is correct (True) or not (False).
        k_values: List/sequence of k's for which pass@k metrics should be computed.
        method: Metric variant to use. Supported:
            - "unbiased": Unbiased estimator for pass@k as defined in
              https://arxiv.org/abs/2108.07732, evaluates probability at least one correct sample in k draws.
            - "first_k": Returns 1.0 if any of the first k generations is correct, 0.0 otherwise.

    Returns:
        A dict mapping e.g. "pass@1" or "pass@5" to their corresponding metric for this sample.
    """
    num_generations = len(correctness_per_generation)
    num_correct = sum(correctness_per_generation)
    pass_at_k_metrics: Dict[str, float] = {}

    for k in k_values:
        # Cannot compute pass@k if fewer than k generations are provided
        if k > num_generations:
            raise ValueError(
                f"k={k} requires at least {k} generations, got {num_generations}"
            )

        if method == "first_k":
            # score is 1.0 if any of the first k generations is correct, 0.0 otherwise
            score = 1.0 if any(correctness_per_generation[:k]) else 0.0
        elif method == "unbiased":
            # Unbiased estimator for pass@k: probability of generating at least one correct sample in k
            if num_correct == 0:
                # No correct generations at all
                score = 0.0
            elif num_generations - num_correct < k:
                # Not enough incorrect generations to choose k wrong ones; all selections will include at least 1 correct
                score = 1.0
            else:
                # Classic formula: 1 - (C(n-c, k) / C(n, k)), n=total generations, c=num correct
                score = 1.0 - math.comb(num_generations - num_correct, k) / math.comb(num_generations, k)
        else:
            raise ValueError(f"Unknown pass@k method: {method}")

        pass_at_k_metrics[f"pass@{k}"] = score

    return pass_at_k_metrics


def evaluate_pass_at_k(
    model_path: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    k_values: Optional[Sequence[int]] = None,
    n_generations: int = 16,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    dtype: str = "bfloat16",
    seed: int = 42,
    method: str = "unbiased",
    show_progress: bool = True,
    question_batch_size: int = 4,
    recipe: Optional[str] = None,
    dataset: str = "gsm8k",
) -> PassAtKResult:
    if k_values is None:
        k_values = [1, 8]
    if n_generations < max(k_values):
        raise ValueError(
            f"n_generations={n_generations} must be >= max(k_values)={max(k_values)}"
        )

    if recipe:
        examples = load_recipe(recipe, max_samples=max_samples, seed=seed)
        dataset_name = recipe
        split_name = recipe
    else:
        examples = load_dataset_by_name(dataset, split=split, max_samples=max_samples, seed=seed)
        dataset_name = dataset
        split_name = split
    loaded = load_model_and_tokenizer(model_path=model_path, dtype=dtype)

    per_question: List[QuestionResult] = []
    aggregate_pass: Dict[str, float] = {f"pass@{k}": 0.0 for k in k_values}

    iterator = examples
    if show_progress:
        iterator = tqdm(examples, desc="Pass@K eval", unit="q")

    for batch_examples in batched(iterator, question_batch_size):
        # prompts = [format_prompt(loaded.tokenizer, ex.messages) for ex in batch_examples]
        _prompts, completions, _, _, _, _ = generate_rollouts(
            loaded=loaded,
            examples=batch_examples,
            n_generations=n_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            tokenize_outputs=False,
        )
        correct_masks = [
            [answers_match(c, ex.answer) for c in c_list]
            for c_list, ex in zip(completions, batch_examples)
        ]
        q_metrics = [
            compute_pass_at_k(cm, k_values, method=method)
            for cm in correct_masks
        ]

        for q_metric in q_metrics:
            for key, value in q_metric.items():
                aggregate_pass[key] += value

        for ex, c_list, cm_list, q_metric in zip(batch_examples, completions, correct_masks, q_metrics):
            per_question.append(
                QuestionResult(
                    example_id=ex.example_id,
                    question=ex.question,
                    ground_truth=ex.answer,
                    completions=c_list,
                    correct_mask=cm_list,
                    num_correct=sum(cm_list),
                )
            )
   

    num_examples = len(per_question)
    for key in aggregate_pass:
        aggregate_pass[key] /= max(num_examples, 1)

    avg_num_correct = sum(q.num_correct for q in per_question) / max(num_examples, 1)

    return PassAtKResult(
        model_path=model_path,
        dataset=dataset_name,
        split=split_name,
        n_generations=n_generations,
        temperature=temperature,
        k_values=list(k_values),
        num_examples=num_examples,
        pass_at_k=aggregate_pass,
        avg_num_correct=avg_num_correct,
        per_question=per_question,
    )


def save_results(result: PassAtKResult, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "pass_at_k_summary.json"
    details_path = out / "pass_at_k_details.jsonl"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_summary_dict(), f, indent=2)

    with details_path.open("w", encoding="utf-8") as f:
        for row in result.per_question:
            payload = asdict(row)
            f.write(json.dumps(payload) + "\n")

    return summary_path
