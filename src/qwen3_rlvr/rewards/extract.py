"""GSM8K answer extraction and equality checks."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

from sympy import sympify
from sympy.core.numbers import Number


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize_numeric(text: str) -> Optional[str]:
    text = text.strip().replace(",", "").replace("$", "").replace("%", "")
    if not text:
        return None
    try:
        value = sympify(text)
        if isinstance(value, Number):
            if value == value.to_integral_value() and abs(value) < 1e15:
                return str(int(value))
            return str(float(value))
    except (TypeError, ValueError, SyntaxError, AttributeError):
        pass
    try:
        frac = Fraction(text)
        if frac.denominator == 1:
            return str(frac.numerator)
        return str(float(frac))
    except (ValueError, ZeroDivisionError):
        return text.strip().lower()


def extract_answer(text: str, style: str = "gsm8k_hash") -> str:
    """Extract a final answer from model output or GSM8K reference."""
    if style == "gsm8k_hash" and "####" in text:
        return text.split("####")[-1].strip()

    matches = _NUM_RE.findall(text.replace(",", ""))
    if matches:
        return matches[-1].strip()

    return text.strip()


def answers_match(prediction: str, reference: str) -> bool:
    """Return True when extracted prediction matches reference answer."""
    pred = extract_answer(prediction)
    ref = extract_answer(reference)

    pred_norm = _normalize_numeric(pred)
    ref_norm = _normalize_numeric(ref)
    if pred_norm is not None and ref_norm is not None:
        return pred_norm == ref_norm

    return pred.strip().lower() == ref.strip().lower()
