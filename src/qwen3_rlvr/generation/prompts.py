"""Chat prompt formatting."""

from __future__ import annotations

from typing import Sequence


def format_prompt(tokenizer, messages: Sequence[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
