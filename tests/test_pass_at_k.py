from qwen3_rlvr.eval.pass_at_k import compute_pass_at_k


def test_compute_pass_at_k_unbiased_all_correct():
    metrics = compute_pass_at_k([True, True, True, True], [1, 2], method="unbiased")
    assert metrics["pass@1"] == 1.0
    assert metrics["pass@2"] == 1.0


def test_compute_pass_at_k_first_k():
    metrics = compute_pass_at_k([False, True, False], [1, 2], method="first_k")
    assert metrics["pass@1"] == 0.0
    assert metrics["pass@2"] == 1.0


def test_compute_pass_at_k_unbiased_none_correct():
    metrics = compute_pass_at_k([False, False, False, False], [1, 4], method="unbiased")
    assert metrics["pass@1"] == 0.0
    assert metrics["pass@4"] == 0.0
