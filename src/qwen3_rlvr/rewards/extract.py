"""Answer extraction and equality checks for verifiable math tasks."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

from sympy import sympify
from sympy.core.numbers import Number

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_HASH_RE = re.compile(r"####\s*(.+?)(?:\n|$)", re.MULTILINE)


def _extract_boxed(text: str) -> Optional[str]:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    idx = start + len(marker)
    depth = 1
    while idx < len(text):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker) : idx].strip()
        idx += 1
    return None


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


def extract_answer(text: str) -> str:
    """Extract a final answer from model output."""
    hash_match = _HASH_RE.search(text)
    if hash_match:
        return hash_match.group(1).strip()

    if "####" in text:
        return text.split("####")[-1].strip()

    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    matches = _NUM_RE.findall(text.replace(",", ""))
    if matches:
        return matches[-1].strip()

    return text.strip()


def extract_reference_answer(text: str, source: str) -> str:
    """Extract canonical reference answers from dataset-specific fields."""
    source = source.lower()
    if source == "gsm8k":
        if "####" in text:
            return text.split("####")[-1].strip()
        return extract_answer(text)

    if source == "math":
        boxed = _extract_boxed(text)
        if boxed is not None:
            return boxed
        return extract_answer(text)

    if source == "aime":
        return str(text).strip()

    return extract_answer(text)


def answers_match(prediction: str, reference: str) -> bool:
    """Return True when extracted prediction matches a canonical reference answer."""
    pred = extract_answer(prediction)
    ref = reference.strip()
    # References usually arrive already canonical, but some callers pass raw
    # "#### <answer>" text; strip that marker so the answer normalizes cleanly.
    if "####" in ref:
        ref = ref.split("####")[-1].strip()

    pred_norm = _normalize_numeric(pred)
    ref_norm = _normalize_numeric(ref)
    if pred_norm is not None and ref_norm is not None:
        return pred_norm == ref_norm

    return pred.strip().lower() == ref.strip().lower()
