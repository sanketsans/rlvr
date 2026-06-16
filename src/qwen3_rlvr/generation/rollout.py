"""Sample completions from the policy for GRPO rollouts."""

from __future__ import annotations

from typing import List, Sequence, Optional

import torch

from qwen3_rlvr.data.gsm8k import Gsm8kExample
from qwen3_rlvr.generation.prompts import format_prompt
from qwen3_rlvr.model.load import LoadedModel
from qwen3_rlvr.rl.grpo import batched_sequence_log_probs


def generate_rollouts(
    loaded: LoadedModel,
    examples: Sequence[Gsm8kExample],
    n_generations: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    return_logprobs: bool = False,
) -> tuple[List[str], List[List[str]], Optional[List[torch.Tensor]]]:
    """Return prompts and completions per example (list of N strings each).
    If return_logprobs is True, return the log probabilities of the completions.
    """
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
    completion_ids = outputs[:, prompt_len:]
    decoded = tokenizer.batch_decode(
        completion_ids,
        skip_special_tokens=True,
    )

    all_completions = [
        decoded[i * n_generations:(i + 1) * n_generations]
        for i in range(batch_size)
    ]

    if return_logprobs:
        with torch.no_grad():
            logprobs = batched_sequence_log_probs(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                completions=all_completions,
                device=device,
            )
        return prompts, all_completions, logprobs
    else:
        return prompts, all_completions
