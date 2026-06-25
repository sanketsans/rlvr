from qwen3_rlvr.rewards.extract import answers_match, extract_answer


def test_extract_hash_answer():
    assert extract_answer("Work #### 42") == "42"
    assert extract_answer("Reasoning\n#### 18") == "18"


def test_answers_match_numeric_variants():
    # `reference` is the canonical answer the dataset loaders already extracted
    # (no "####" prefix); only the prediction is raw model output.
    assert answers_match("#### 42", "42.0")
    assert answers_match("The answer is 1,234", "1234")
    assert not answers_match("#### 41", "42")
