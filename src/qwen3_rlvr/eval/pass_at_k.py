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

from qwen3_rlvr.data.gsm8k import Gsm8kExample, load_gsm8k
from qwen3_rlvr.model.load import LoadedModel, load_model_and_tokenizer
from qwen3_rlvr.rewards.extract import answers_match, extract_answer


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

def _format_prompt(tokenizer, messages: Sequence[dict]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _generate_completions(
    loaded: LoadedModel,
    prompts: List[str],
    n_generations: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> List[List[str]]:
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = loaded.device
    batch_size = len(prompts)
    torch.manual_seed(seed)

    inputs = tokenizer(
        prompts,                    # List[str]
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    input_len = inputs["input_ids"].shape[-1]

    do_sample = temperature > 0
    gen_kwargs = {
        "num_return_sequences": n_generations,
        # "temperature": temperature, # pass temp only when do_sample=True
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )

    completions = []

    for prompt_idx in range(batch_size):
        prompt_completions = []
        for sample_idx in range(n_generations):
            row = prompt_idx * n_generations + sample_idx
            text = tokenizer.decode(
                outputs[row, input_len:],
                skip_special_tokens=True,
            ).strip()
            prompt_completions.append(text)
        completions.append(prompt_completions)

    return completions


def compute_pass_at_k(correct_mask: Sequence[bool], k_values: Sequence[int], method: str = "unbiased") -> Dict[str, float]:
    n = len(correct_mask)
    c = sum(correct_mask)
    metrics: Dict[str, float] = {}

    for k in k_values:
        if k > n:
            raise ValueError(f"k={k} requires at least {k} generations, got {n}")

        if method == "first_k":
            score = 1.0 if any(correct_mask[:k]) else 0.0
        elif method == "unbiased":
            if c == 0:
                score = 0.0
            elif n - c < k:
                score = 1.0
            else:
                score = 1.0 - math.comb(n - c, k) / math.comb(n, k)
        else:
            raise ValueError(f"Unknown pass@k method: {method}")

        metrics[f"pass@{k}"] = score
    return metrics


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
) -> PassAtKResult:
    if k_values is None:
        k_values = [1, 8, 16]
    if n_generations < max(k_values):
        raise ValueError(
            f"n_generations={n_generations} must be >= max(k_values)={max(k_values)}"
        )

    examples = load_gsm8k(split=split, max_samples=max_samples, seed=seed)
    loaded = load_model_and_tokenizer(model_path=model_path, dtype=dtype)

    per_question: List[QuestionResult] = []
    aggregate_pass: Dict[str, float] = {f"pass@{k}": 0.0 for k in k_values}

    iterator = examples
    if show_progress:
        iterator = tqdm(examples, desc="Pass@K eval", unit="q")

    for batch_examples in batched(iterator, question_batch_size):
        prompts = [_format_prompt(loaded.tokenizer, ex.messages) for ex in batch_examples]
        completions = _generate_completions(
            loaded=loaded,
            prompts=prompts,
            n_generations=n_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
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
                    ground_truth=extract_answer(ex.answer),
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
        dataset="gsm8k",
        split=split,
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
