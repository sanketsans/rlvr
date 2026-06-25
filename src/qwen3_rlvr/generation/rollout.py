"""Sample completions from the policy for GRPO rollouts."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from qwen3_rlvr.data.base import VerifiableExample
from qwen3_rlvr.generation.prompts import format_prompt
from qwen3_rlvr.model.load import LoadedModel
from qwen3_rlvr.rl.grpo import batched_sequence_log_probs, batched_tokenize_prompt_completion


def _decode_generated_batch(tokenizer, outputs, prompt_len, batch_size, n_generations):
    """Decode a flat ``generate()`` output tensor into per-prompt completion lists.

    ``outputs`` has shape ``(batch_size * n_generations, seq_len)``; the first
    ``prompt_len`` tokens are the prompt and are stripped before decoding. The
    flat list of decoded strings is then regrouped so result[i] holds the
    ``n_generations`` completions for prompt ``i``.
    """
    completion_ids = outputs[:, prompt_len:]
    decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return [decoded[i * n_generations : (i + 1) * n_generations] for i in range(batch_size)]


def generate_rollouts(
    loaded: LoadedModel,
    examples: Sequence[VerifiableExample],
    n_generations: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    return_logprobs: bool = False,
    tokenize_outputs: bool = True,
) -> tuple[
    List[str],
    List[List[str]],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Return prompts and completions per example (list of N strings each).
    If return_logprobs is True, return the log probabilities of the completions.
    Set tokenize_outputs=False for eval-only generation to avoid extra GPU tensors.
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
    del inputs
    if was_training:
        model.train()

    batch_size = len(prompts)
    all_completions = _decode_generated_batch(
        tokenizer, outputs, prompt_len, batch_size, n_generations
    )
    del outputs

    if not tokenize_outputs:
        return prompts, all_completions, None, None, None, None

    tokenized_input_ids, tokenized_attention_mask, tokenized_completion_mask = (
        batched_tokenize_prompt_completion(tokenizer, prompts, all_completions, device)
    )
    if return_logprobs:
        with torch.no_grad():
            logprobs = batched_sequence_log_probs(
                model=model,
                tokenized_input_ids=tokenized_input_ids,
                tokenized_attention_mask=tokenized_attention_mask,
                tokenized_completion_mask=tokenized_completion_mask,
            )
        return (
            prompts,
            all_completions,
            tokenized_input_ids,
            tokenized_attention_mask,
            tokenized_completion_mask,
            logprobs,
        )
    return (
        prompts,
        all_completions,
        tokenized_input_ids,
        tokenized_attention_mask,
        tokenized_completion_mask,
        None,
    )
