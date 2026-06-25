from qwen3_rlvr.rewards.extract import (
    _extract_boxed,
    _normalize_numeric,
    answers_match,
    extract_answer,
    extract_reference_answer,
)


def test_extract_boxed_simple_and_nested():
    assert _extract_boxed(r"the answer is \boxed{42}") == "42"
    # Brace matching should keep the full nested expression.
    assert _extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_boxed_missing_returns_none():
    assert _extract_boxed("no box here") is None


def test_extract_answer_prefers_hash_over_trailing_number():
    assert extract_answer("we computed 7 along the way\n#### 12") == "12"


def test_extract_answer_falls_back_to_last_number():
    assert extract_answer("first 3 then 4 then 72") == "72"


def test_normalize_numeric_equates_variants():
    assert _normalize_numeric("42.0") == _normalize_numeric("42") == "42"
    assert _normalize_numeric("1,234") == "1234"
    assert _normalize_numeric("1/2") == _normalize_numeric("0.5")


def test_extract_reference_answer_by_source():
    assert extract_reference_answer("blah #### 7", "gsm8k") == "7"
    assert extract_reference_answer(r"\boxed{9}", "math") == "9"
    assert extract_reference_answer("  17 ", "aime") == "17"


def test_answers_match_handles_fraction_and_decimal():
    assert answers_match(r"so \boxed{0.5}", "1/2")


def test_answers_match_non_numeric_falls_back_to_string():
    # With no number/box to extract, the prediction is compared as-is,
    # case-insensitively, against the reference.
    assert answers_match("Blue", "blue")
    assert not answers_match("Red", "blue")
