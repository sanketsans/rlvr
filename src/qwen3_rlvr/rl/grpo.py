"""GRPO advantage computation and policy loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import numpy as np 


@dataclass
class GRPOBatch:
    rewards: torch.Tensor  # [B, N_generations]
    advantages: torch.Tensor  # [B, N_generations]
    old_token_logp: torch.Tensor  # [B*N_generations, L_prompt + L_completion - 1] # because they are shifted by 1 since causal. 
    tokenized_input_ids: torch.Tensor  # [B*N_generations, L_prompt + L_completion]
    tokenized_attention_mask: torch.Tensor  # [B*N_generations, L_prompt + L_completion]
    tokenized_completion_mask: torch.Tensor  # [B*N_generations, L_prompt + L_completion]


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
    # was previosly doing the full text (prompt + completion) tokenization
    # that can cause issues, since BPE tokenization works with neighboring tokens.
    # so, if we combined prompts + completions, the prefix/ suffix tokens of completions and prompts respectively would be tokenized differently.
    # this is not desired, since we want to compute the log-probs for the completion tokens only.
    # so, we tokenize the prompt and completion separately, and then combine them.
    # this ensures that the prefix/ suffix tokens of completions and prompts respectively are tokenized the same.
    tokenizer.padding_side = "right"
    prompts = np.repeat(prompts, len(completions[0]), axis=0).tolist()
    completions = np.concatenate(completions, axis=0).tolist()  # assuming each prompt has the same number of completions.

    assert len(prompts) == len(completions)  
    full_text_input_ids = []
    full_text_attention_mask = []

    prompt_enc = tokenizer(prompts, add_special_tokens=True, padding=False, truncation=True)
    completion_enc = tokenizer(completions, add_special_tokens=False, padding=False, truncation=True)

    for prompt_ids, completion_ids in zip(prompt_enc['input_ids'], completion_enc['input_ids']):
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

    input_ids = pad_sequence(full_text_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    completion_mask = pad_sequence(full_text_attention_mask, batch_first=True, padding_value=0)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return (
        input_ids.to(device), 
        attention_mask.to(device), 
        completion_mask.to(device)
    )

def batched_sequence_log_probs(
    model: nn.Module,
    tokenized_input_ids, 
    tokenized_attention_mask,
    tokenized_completion_mask,
) -> torch.Tensor:
    """Per-sequence sum of log-probs over completion tokens. Returns shape [B]."""
    # outputs: logits for all tokens in the input_ids - policy gradient should happen on prompt + completion tokens, not just on completion tokens
    # attention mask : 1 denotes to be included in the log-probs, 0 denotes to be ignored. But shape of output remain same as input_ids. 
    outputs = model(input_ids=tokenized_input_ids, attention_mask=tokenized_attention_mask)
    logits = outputs.logits[:, :-1, :] # logits for all tokens in the input_ids - policy gradient should happen on prompt + completion tokens, not just on completion tokens
    targets = tokenized_input_ids[:, 1:] # targets are shifted by 1 since causal. 
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # Typically, tokenized_completion_mask[:, 1:] is used to ignore the first token (which may be a BOS token or start of the completion), 
    # but whether you need to do this depends on what logprobs and masks represent in your specific pipeline.
    # If you want to apply 'valid_mask' to select only those tokens that are actually part of the completion,
    # dropping the initial token may be necessary (for autoregressive models, the first token's logprob may be undefined).
    # If unsure, consider inspecting shapes and intended downstream usage.
    tokenizer_completion_mask = tokenized_completion_mask[:, 1:] # [n_prompts * n_generations, n_tokens_in_completion]
    return token_log_probs * tokenizer_completion_mask # [n_prompts * n_generations, n_tokens_in_completion]

def compute_policy_loss(
    policy: nn.Module,
    tokenizer,
    reference: nn.Module,
    grpo_batch: GRPOBatch,
    kl_coef: Optional[float] = 0.5,
    reinforce: bool = False,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Aggregate GRPO loss over all question-completion pairs."""
    policy.train()
    reference.eval()
    tokenized_input_ids = grpo_batch.tokenized_input_ids
    tokenized_attention_mask = grpo_batch.tokenized_attention_mask
    tokenized_completion_mask = grpo_batch.tokenized_completion_mask
    advantages = grpo_batch.advantages.reshape(-1) # [B, N] -> [B*N_generations]

    active_advantage_mask = advantages.abs() > 1e-8
    if active_advantage_mask.sum() == 0:
        return torch.zeros((), device=advantages.device), {"num_loss_terms": 0, "policy_logp_mean": 0.0, "kl_mean": 0.0} 

    # since tokens probs only on the logs 
    # shape is [B*N_generations, L_prompt + L_completion - 1]
    policy_token_logp = batched_sequence_log_probs(
                                policy, 
                                tokenized_input_ids, 
                                tokenized_attention_mask, 
                                tokenized_completion_mask
                                )
    policy_token_logp = policy_token_logp[active_advantage_mask]
    advantages = advantages[active_advantage_mask].to(policy_token_logp.device)

    if reinforce: 
        sequence_log_probs = policy_token_logp.sum(dim=1) 
        loss = -(sequence_log_probs * advantages).mean()
        return loss, {"num_loss_terms": active_advantage_mask.sum().item(), "policy_logp_mean": sequence_log_probs.mean().item(), "advantage_mean": advantages.mean().item()}

    old_token_logp = grpo_batch.old_token_logp[active_advantage_mask]
    with torch.no_grad():
        reference_token_logp = batched_sequence_log_probs(
                                reference, 
                                tokenized_input_ids, 
                                tokenized_attention_mask, 
                                tokenized_completion_mask
                                )

        reference_token_logp = reference_token_logp[active_advantage_mask]

    # PPO ratio 

    # TODO: check the shapes here 
    advantages = advantages.unsqueeze(-1) 
    # only valid tokens contribute 
    # Typically, tokenized_completion_mask[:, 1:] to align with token log probs shape.
    # since we are only optimizing for next-token. current completion mask is for all the tokens. 
    # that's why we move by 1 to match the shape of token log probs.
    valid_mask = tokenized_completion_mask[:, 1:].float()[active_advantage_mask]
    ratio = torch.exp(policy_token_logp - old_token_logp)
    clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_token_loss = -torch.min(ratio * advantages, clip_adv)

    # Deepseek GRPO objective 
    delta = reference_token_logp - policy_token_logp 
    kl = (torch.exp(delta) - delta - 1)

    pg_loss = (pg_token_loss * valid_mask).mean()
    kl_loss = (kl * valid_mask).mean() 
    loss = pg_loss + kl_coef * kl_loss

    clip_fraction = (
        ((ratio - 1.0).abs() > clip_eps)
        * valid_mask.bool()
    ).float().sum() / valid_mask.sum()

    return loss, {
        "num_loss_terms": active_advantage_mask.sum().item(),
        "policy_token_logp_mean": policy_token_logp.mean().item(),
        "reference_token_logp_mean": reference_token_logp.mean().item(),
        "pg_loss": pg_loss.item(),
        "kl_loss": kl_loss.item(),
        "clip_fraction": clip_fraction.item(),
        "ratio_mean": ratio.mean().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
    }