import torch

from qwen3_rlvr.rl.grpo import compute_advantages


def test_advantages_are_group_normalized():
    # Two questions (rows), each with 4 completions; advantages normalize per row.
    rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    adv = compute_advantages(rewards)
    # Each row should be ~zero-mean after group normalization.
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-5)
    assert adv.shape == rewards.shape


def test_uniform_rewards_give_zero_advantage():
    # No spread within a group -> nothing to learn -> advantages ~0.
    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    adv = compute_advantages(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-4)


def test_single_generation_returns_zeros():
    # With one completion per question there is no group to normalize against.
    rewards = torch.tensor([[1.0], [0.0]])
    adv = compute_advantages(rewards)
    assert torch.equal(adv, torch.zeros_like(rewards))


def test_higher_reward_gets_higher_advantage():
    rewards = torch.tensor([[0.0, 1.0]])
    adv = compute_advantages(rewards)
    assert adv[0, 1] > adv[0, 0]
