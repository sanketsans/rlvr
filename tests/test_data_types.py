from qwen3_rlvr.data.base import SFTExample, VerifiableExample


def test_verifiable_example_messages_are_system_then_user():
    ex = VerifiableExample(example_id=0, question="2+2?", answer="4", source="gsm8k")
    msgs = ex.messages
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "2+2?"
    # The system prompt is selected by source.
    assert "#### <answer>" in msgs[0]["content"]


def test_verifiable_example_system_prompt_varies_by_source():
    gsm8k = VerifiableExample(0, "q", "a", "gsm8k").messages[0]["content"]
    math = VerifiableExample(0, "q", "a", "math").messages[0]["content"]
    assert gsm8k != math
    assert r"\boxed{}" in math


def test_sft_example_messages_include_assistant_completion():
    ex = SFTExample(example_id=1, question="q", completion="the work\n#### 5", source="gsm8k")
    msgs = ex.messages
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2]["content"] == "the work\n#### 5"
