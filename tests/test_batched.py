from qwen3_rlvr.eval.pass_at_k import batched


def test_batched_even_split():
    assert list(batched([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_batched_remainder_in_last_chunk():
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_batched_size_larger_than_iterable():
    assert list(batched([1, 2], 10)) == [[1, 2]]


def test_batched_empty_iterable():
    assert list(batched([], 3)) == []
