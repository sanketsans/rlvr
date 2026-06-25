import pytest

from qwen3_rlvr.data.recipe import (
    _allocate_counts,
    list_recipes,
    load_dataset_by_name,
    load_recipe,
)


def test_allocate_counts_splits_by_weight():
    # Equal weights -> even split that sums to the total.
    counts = _allocate_counts(10, [1.0, 1.0])
    assert counts == [5, 5]
    assert sum(counts) == 10


def test_allocate_counts_distributes_remainder_to_largest_fraction():
    # 10 split 1:3 -> 2.5 / 7.5; remainder lands so the total is preserved.
    counts = _allocate_counts(10, [1.0, 3.0])
    assert sum(counts) == 10
    assert counts == [3, 7] or counts == [2, 8]


def test_allocate_counts_zero_total():
    assert _allocate_counts(0, [1.0, 2.0]) == [0, 0]


def test_allocate_counts_nonpositive_weights_raise():
    with pytest.raises(ValueError):
        _allocate_counts(10, [0.0, 0.0])


def test_list_recipes_includes_known_recipes():
    recipes = list_recipes()
    assert "gsm8k_train" in recipes
    assert "gsm8k_test" in recipes
    assert recipes == sorted(recipes)


def test_load_recipe_unknown_raises():
    with pytest.raises(ValueError, match="Unknown recipe"):
        load_recipe("does_not_exist")


def test_load_dataset_by_name_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset_by_name("does_not_exist")
