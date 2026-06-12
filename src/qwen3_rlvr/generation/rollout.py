"""Sample completions from the policy for GRPO rollouts."""

from __future__ import annotations

from typing import List, Sequence

import torch

from qwen3_rlvr.data.gsm8k import Gsm8kExample
from qwen3_rlvr.generation.prompts import format_prompt
from qwen3_rlvr.model.load import LoadedModel


def generate_rollouts(
    loaded: LoadedModel,
    examples: Sequence[Gsm8kExample],
    n_generations: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[List[str], List[List[str]]]:
    """Return prompts and completions per example (list of N strings each)."""
    tokenizer = loaded.tokenizer
    tokenizer.padding_side = "left"
    model = loaded.model
    device = loaded.device

    prompts = [format_prompt(tokenizer, ex.messages) for ex in examples]
    torch.manual_seed(seed)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    if not do_sample and n_generations > 1:
        raise ValueError("n_generations > 1 requires temperature > 0")

    gen_kwargs = {
        "num_return_sequences": n_generations,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    if was_training:
        model.train()

    batch_size = len(prompts)
    all_completions: List[List[str]] = []
    for prompt_idx in range(batch_size):
        texts: List[str] = []
        for sample_idx in range(n_generations):
            row = prompt_idx * n_generations + sample_idx
            text = tokenizer.decode(outputs[row, prompt_len:], skip_special_tokens=True).strip()
            texts.append(text)
        all_completions.append(texts)

    return prompts, all_completions
