"""Model and tokenizer loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


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


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "bfloat16",
    device: Optional[str] = None,
) -> LoadedModel:
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    resolved = next(model.parameters()).device
    return LoadedModel(model=model, tokenizer=tokenizer, device=resolved)
