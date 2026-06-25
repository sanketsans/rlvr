from qwen3_rlvr.generation.rollout import _decode_generated_batch


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(t) + ord("a")) for t in token_ids if int(t) > 0)

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]


def test_decode_generated_batch():
    # batch=2, n_gen=2, prompt_len=3, gen len=2 each -> total 5
    outputs = __import__("torch").tensor(
        [
            [0, 0, 1, 10, 11],
            [0, 0, 1, 12, 13],
            [0, 2, 3, 14, 15],
            [0, 2, 3, 16, 17],
        ]
    )
    decoded = _decode_generated_batch(_FakeTokenizer(), outputs, prompt_len=3, batch_size=2, n_generations=2)
    assert len(decoded) == 2
    assert len(decoded[0]) == 2
    assert decoded[0][0] == "kl"
