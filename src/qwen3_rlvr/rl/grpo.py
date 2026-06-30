"""GRPO advantage computation and policy loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence


@dataclass
class GRPOBatch:
    rewards: torch.Tensor  # [B, N_generations]
    advantages: torch.Tensor  # [B, N_generations]
    tokenized_input_ids: Optional[torch.Tensor]  # [B*N_generations, L_prompt + L_completion]
    tokenized_attention_mask: Optional[torch.Tensor]  # [B*N_generations, L_prompt + L_completion]
    tokenized_completion_mask: Optional[torch.Tensor]  # [B*N_generations, L_prompt + L_completion]
    # Optional: not set by the RFT trainer, which does not compute old log-probs.
    old_token_logp: Optional[torch.Tensor] = (
        None  # [B*N_generations, L-1]; causal shift drops one position
    )


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Group-normalize rewards per question (dim=1)."""
    mean = rewards.mean(dim=1, keepdim=True)
    if rewards.shape[1] == 1:
        return torch.zeros_like(rewards)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def batched_tokenize_prompt_completion(
    tokenizer,
    prompts: List[str],
    completions: List[List[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Tokenize prompt and completion separately, then concatenate the token ids.
    # Tokenizing the joined string instead would let BPE merge tokens across the
    # prompt/completion boundary, shifting which tokens belong to the completion
    # and corrupting the per-completion log-probs we compute downstream.
    tokenizer.padding_side = "right"
    prompts = np.repeat(prompts, len(completions[0]), axis=0).tolist()
    completions = np.concatenate(
        completions, axis=0
    ).tolist()  # assuming each prompt has the same number of completions.

    assert len(prompts) == len(completions)
    full_text_input_ids = []
    full_text_attention_mask = []

    prompt_enc = tokenizer(prompts, add_special_tokens=True, padding=False, truncation=True)
    completion_enc = tokenizer(
        completions, add_special_tokens=False, padding=False, truncation=True
    )

    for prompt_ids, completion_ids in zip(prompt_enc["input_ids"], completion_enc["input_ids"]):
        # prompt_ids: Tensor [L_prompt], completion_ids: Tensor [L_completion]
        prompt_ids = torch.tensor(prompt_ids)
        completion_ids = torch.tensor(completion_ids)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=0)  # [L_prompt + L_completion]
        sequence_length = full_ids.shape[0]
        full_text_input_ids.append(full_ids)
        mask = torch.zeros(sequence_length, dtype=torch.bool)  # [L_prompt + L_completion]
        prompt_len = prompt_ids.shape[0]
        mask[prompt_len:] = True
        full_text_attention_mask.append(mask)

    input_ids = pad_sequence(
        full_text_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    completion_mask = pad_sequence(full_text_attention_mask, batch_first=True, padding_value=0)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return (input_ids.to(device), attention_mask.to(device), completion_mask.to(device))


def batched_sequence_log_probs(
    model: nn.Module,
    tokenized_input_ids,
    tokenized_attention_mask,
    tokenized_completion_mask,
) -> torch.Tensor:
    """Per-token log-probs over the sequence, zeroed outside completion tokens.

    Returns shape [B, L-1]: the log-prob the model assigns to each next token.
    """
    outputs = model(
        input_ids=tokenized_input_ids,
        attention_mask=tokenized_attention_mask,
    )
    # Causal shift: logits at position t predict the token at position t+1, so we
    # drop the last logit and the first input id to line predictions up with targets.
    # Upcast the LM-head logits to fp32 before the softmax. Following the Kimi K
    # report, computing the final logits/log-softmax in fp32 avoids the bf16 rounding
    # over the large vocab that biases per-token log-probs and  breakpoint()  # inspect outputs.logits dtype here: compare .float() vs raw fp16 head
    logits = outputs.logits[:, :-1, :].float()
    targets = tokenized_input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Slice the completion mask the same way so it aligns with the shifted targets,
    # then zero out prompt tokens — only completion tokens carry gradient.
    tokenizer_completion_mask = tokenized_completion_mask[:, 1:]  # [B, L-1]
    return token_log_probs * tokenizer_completion_mask  # [B, L-1]


def compute_policy_loss(
    policy: nn.Module,
    reference: nn.Module,
    grpo_batch: GRPOBatch,
    tokenizer: Optional[Any] = None,
    kl_coef: float = 0.02,
    reinforce: bool = False,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Aggregate GRPO loss over all question-completion pairs."""
    policy.train()
    reference.eval()
    assert grpo_batch.old_token_logp is not None
    assert grpo_batch.tokenized_attention_mask is not None
    assert grpo_batch.tokenized_completion_mask is not None
    assert grpo_batch.tokenized_input_ids is not None
    tokenized_input_ids = grpo_batch.tokenized_input_ids
    tokenized_attention_mask = grpo_batch.tokenized_attention_mask
    tokenized_completion_mask = grpo_batch.tokenized_completion_mask
    advantages = grpo_batch.advantages.reshape(-1)  # [B, N] -> [B*N_generations]

    active_advantage_mask = advantages.abs() > 1e-8
    if active_advantage_mask.sum() == 0:
        return torch.zeros((), device=advantages.device), {
            "num_loss_terms": 0,
            "policy_logp_mean": 0.0,
            "kl_mean": 0.0,
        }

    # Per-token log-probs under the current policy; shape [B*N, L-1].
    policy_token_logp = batched_sequence_log_probs(
        policy, tokenized_input_ids, tokenized_attention_mask, tokenized_completion_mask
    )
    policy_token_logp = policy_token_logp[active_advantage_mask].float()
    advantages = advantages[active_advantage_mask].float().to(policy_token_logp.device)

    if reinforce:
        sequence_log_probs = policy_token_logp.sum(dim=1)
        loss = -(sequence_log_probs * advantages).mean()
        return loss, {
            "num_loss_terms": active_advantage_mask.sum().item(),
            "policy_logp_mean": sequence_log_probs.mean().item(),
            "advantage_mean": advantages.mean().item(),
        }
    old_token_logp = grpo_batch.old_token_logp[active_advantage_mask].float()
    with torch.no_grad():
        reference_token_logp = batched_sequence_log_probs(
            reference, tokenized_input_ids, tokenized_attention_mask, tokenized_completion_mask
        )

        reference_token_logp = reference_token_logp[active_advantage_mask].float()

    # PPO-style clipped surrogate, per token. advantages -> [N, 1] to broadcast over length.
    advantages = advantages.unsqueeze(-1)
    # Slice the completion mask by [:, 1:] to match the causal-shifted log-prob length,
    # so only completion tokens contribute to the loss.
    valid_mask = tokenized_completion_mask[:, 1:].float()[active_advantage_mask]
    ratio = torch.exp(policy_token_logp - old_token_logp)
    clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_token_loss = -torch.min(ratio * advantages, clip_adv)

    # KL penalty to the frozen reference (DeepSeek GRPO's k3 estimator).
    delta = reference_token_logp - policy_token_logp
    kl = torch.exp(delta) - delta - 1

    pg_loss = (pg_token_loss * valid_mask).mean()
    kl_loss = (kl * valid_mask).mean()
    loss = pg_loss + kl_coef * kl_loss

    clip_fraction = (
        ((ratio - 1.0).abs() > clip_eps) * valid_mask.bool()
    ).float().sum() / valid_mask.sum()

    return loss, {
        "num_loss_terms": active_advantage_mask.sum().item(),
        "policy_token_logp_mean": policy_token_logp.mean().item(),
        "reference_token_logp_mean": reference_token_logp.mean().item(),
        "pg_loss": pg_loss.item(),
        "kl_loss": kl_loss.item(),
        "clip_fraction": clip_fraction.item(),
        "ratio_mean": ratio.mean().item(),
        "advantage_abs_mean": advantages.abs().mean().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        "delta_mean": delta.mean().item(),
        "delta_std": delta.std().item(),
    }
