from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qwen3_rlvr.rl.grpo import batched_sequence_log_probs, batched_tokenize_prompt_completion
from qwen3_rlvr.rl.mopd import MOPDBatch, compute_mopd_loss, compute_token_advantages


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_single_active_token_normalizes_to_finite_zero(dtype):
    teacher = torch.tensor([[-0.25, 0.0]], dtype=dtype, requires_grad=True)
    prox = torch.tensor([[-1.25, 0.0]], dtype=dtype, requires_grad=True)
    active = torch.tensor([[True, False]])

    advantages, reverse_kl = compute_token_advantages(teacher, prox, active, normalize=True)

    assert torch.isfinite(advantages).all()
    assert torch.equal(advantages, torch.zeros_like(teacher))
    assert torch.equal(reverse_kl, torch.tensor([[-1.0, 0.0]], dtype=dtype))
    assert not advantages.requires_grad
    assert not reverse_kl.requires_grad


def test_multiple_active_tokens_keep_sample_standard_deviation_and_mask():
    teacher = torch.tensor([[-1.0, -0.5, 0.0]])
    prox = torch.tensor([[-2.0, -3.5, 0.0]])
    active = torch.tensor([[True, True, False]])

    advantages, _ = compute_token_advantages(teacher, prox, active, normalize=True)

    # Raw advantages [1, 3] have mean 2 and sample standard deviation sqrt(2).
    expected = torch.tensor([[-0.7071067812, 0.7071067812, 0.0]])
    torch.testing.assert_close(advantages, expected)


def test_normalization_off_preserves_single_token_signal():
    teacher = torch.tensor([[-0.25, 0.0]])
    prox = torch.tensor([[-1.25, 0.0]])
    active = torch.tensor([[True, False]])

    advantages, _ = compute_token_advantages(teacher, prox, active)

    assert torch.equal(advantages, torch.tensor([[1.0, 0.0]]))


def test_no_active_tokens_remain_zero():
    logp = torch.zeros((2, 3))
    active = torch.zeros_like(logp, dtype=torch.bool)

    advantages, _ = compute_token_advantages(logp, logp, active, normalize=True)

    assert torch.equal(advantages, logp)


class _OneTokenTokenizer:
    pad_token_id = 0

    def __call__(self, texts, *, add_special_tokens, **kwargs):
        token = 1 if add_special_tokens else 2
        return {"input_ids": [[token] for _ in texts]}


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3))

    def forward(self, input_ids, attention_mask):
        batch_size, length = input_ids.shape
        return SimpleNamespace(logits=self.logits.expand(batch_size, length, -1))


@pytest.mark.parametrize("normalize", [False, True])
def test_single_completion_token_loss_and_backward(normalize):
    # Exercise the actual tokenizer/mask producer and policy log-prob path.
    ids, attention, completion = batched_tokenize_prompt_completion(
        _OneTokenTokenizer(), ["prompt"], [["answer"]], torch.device("cpu")
    )
    policy = _TinyPolicy()
    behavior = batched_sequence_log_probs(policy, ids, attention, completion).detach()
    batch = MOPDBatch(
        tokenized_input_ids=ids,
        tokenized_attention_mask=attention,
        tokenized_completion_mask=completion,
        behavior_token_logp=behavior,
        teacher_token_logp=behavior + 2.0,
    )

    loss, metrics = compute_mopd_loss(policy, batch, normalize_advantages=normalize)
    loss.backward()

    assert metrics["num_completion_tokens"] == 1
    assert torch.isfinite(loss)
    assert torch.isfinite(policy.logits.grad).all()
    if normalize:
        assert loss.item() == 0.0
        assert torch.equal(policy.logits.grad, torch.zeros(3))
    else:
        assert loss.item() == -2.0
        torch.testing.assert_close(
            policy.logits.grad, torch.tensor([2.0 / 3.0, 2.0 / 3.0, -4.0 / 3.0])
        )
