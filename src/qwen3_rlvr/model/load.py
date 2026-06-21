"""Model and tokenizer loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from qwen3_rlvr.logging.logger import setup_logger

logger = setup_logger(__name__)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def _resolve_device(device: Optional[str] = None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "bfloat16",
    device: Optional[str] = None,
    train: bool = False,
) -> LoadedModel:
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
    device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if train:
        model.train()
    else:
        model.eval()

    resolved = next(model.parameters()).device
    logger.info(f"Loaded model on device: {resolved}")
    return LoadedModel(model=model, tokenizer=tokenizer, device=resolved)


def load_policy_and_reference(
    model_path: str,
    dtype: str = "bfloat16",
    device: Optional[str] = None,
) -> Tuple[LoadedModel, LoadedModel]:
    """Load trainable policy and frozen reference with identical init weights."""
    device = _resolve_device(device)
    policy = load_model_and_tokenizer(model_path, dtype=dtype, device=device, train=True)
    reference = load_model_and_tokenizer(model_path, dtype=dtype, device=device, train=False)
    logger.info(f"Loaded policy and reference models on devices: {policy.device} and {reference.device}")
    for param in reference.model.parameters():
        param.requires_grad = False
    logger.info(f"Set reference model parameters to not require gradients")
    return policy, reference
