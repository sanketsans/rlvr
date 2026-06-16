"""GRPO advantage computation and policy loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np 


@dataclass
class GRPOBatch:
    prompts: List[str]
    completions: List[List[str]]
    rewards: torch.Tensor  # [B, N]
    advantages: torch.Tensor  # [B, N]
    old_logprobs: Optional[List[torch.Tensor]] = None


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
    
    full_text = [
        prompt + completion for prompt, completion in zip(prompts, completions)
    ]
    prompt_enc = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True) # [B, L]
    prompt_len = prompt_enc.attention_mask.sum(dim=1).to(device) # [B]
    full_text_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True) # [B, L] 
    input_ids = full_text_enc.input_ids.to(device) 
    attention_mask = full_text_enc.attention_mask.to(device) 

    return input_ids, attention_mask, prompt_len

def batched_sequence_log_probs(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    completions:  List[List[str]],
    device: torch.device,
) -> torch.Tensor:
    """Per-sequence sum of log-probs over completion tokens. Returns shape [B]."""
    tokenizer.padding_side = "right"
    prompts = np.repeat(prompts, len(completions[0]), axis=0).tolist()
    completions = np.concatenate(completions, axis=0).tolist()

    assert len(prompts) == len(completions)  
    # input_ids: prompt tokens + completion tokens (only valid tokens to be included in the log-probs, no padding)
    input_ids, attention_mask, prompt_len = batched_tokenize_prompt_completion(tokenizer, prompts, completions, device)
    # outputs: logits for all tokens in the input_ids - policy gradient should happen on prompt + completion tokens, not just on completion tokens
    # attention mask : 1 denotes to be included in the log-probs, 0 denotes to be ignored. But shape of output remain same as input_ids. 
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :] # logits for all tokens in the input_ids - policy gradient should happen on prompt + completion tokens, not just on completion tokens
    targets = input_ids[:, 1:] # targets are shifted by 1 since causal. 
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # again shifted by 1 since causal. 
    valid_mask = attention_mask[:, 1:].bool()
    positions = torch.arange(token_log_probs.shape[1], device=device).unsqueeze(0)
    completion_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    completion_mask = positions >= (prompt_len - 1).unsqueeze(1)
    completion_mask = completion_mask & valid_mask
    completion_log_probs = token_log_probs * completion_mask
    return completion_log_probs.sum(dim=1)

def compute_policy_loss(
    policy: nn.Module,
    reference: nn.Module,
    tokenizer,
    grpo_batch: GRPOBatch,
    kl_coef: float,
    device: torch.device,
    reinforce: bool = False,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Aggregate GRPO loss over all question-completion pairs."""
    policy.train()
    reference.eval()
    advantages = grpo_batch.advantages.reshape(-1).to(device)

    active_advantage_mask = advantages.abs() > 1e-8
    if active_advantage_mask.sum() == 0:
        return torch.zeros((), device=device), {"num_loss_terms": 0, "policy_logp_mean": 0.0, "kl_mean": 0.0} 

    policy_logp = batched_sequence_log_probs(policy, tokenizer, grpo_batch.prompts, grpo_batch.completions, device) 
    with torch.no_grad():
        reference_logp = batched_sequence_log_probs(reference, tokenizer, grpo_batch.prompts, grpo_batch.completions, device)
        reference_logp = reference_logp[active_advantage_mask] # [B] 
    policy_logp = policy_logp[active_advantage_mask] # [B] 
    advantages = advantages[active_advantage_mask] # [B] 

    if reinforce:
        # REINFORCE 
        loss = (- advantages * policy_logp).mean()
        metrics = {
            "num_loss_terms": active_advantage_mask.sum().item(),
            "policy_logp_mean": policy_logp.mean().item(),
            "kl_mean": 0.0,
        }
        return loss, metrics

    old_log_probs = grpo_batch.old_logprobs.reshape(-1)[active_advantage_mask]
    ratio = torch.exp(policy_logp - old_log_probs) 
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) # [B]

    pg_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages,
    )
    delta = policy_logp - reference_logp # [B]
    kl_term = (torch.exp(delta) - delta - 1.0).mean() # [B]
    loss = (pg_loss + kl_coef * kl_term).mean() # [B]
    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean() # [B]
    return loss, {
        "num_loss_terms": active_advantage_mask.sum().item(),
        "policy_logp_mean": policy_logp.mean().item(),
        "kl_mean": kl_term.mean().item(),
        "clip_fraction": clip_fraction.item(),
    }