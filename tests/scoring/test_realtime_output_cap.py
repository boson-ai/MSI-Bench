"""A reasoning Realtime target gets room for its reasoning; the 700-token default stays for the rest."""

from ib.scoring.openai_fullduplex import _max_output_tokens


def test_medium_and_up_reasoning_get_the_large_cap():
    # gpt-realtime-2.1 medium on MSI-Bench v2 hit 700 output tokens mid-answer on multi-tool cases (status incomplete)
    assert [_max_output_tokens(e) for e in ("medium", "high", "xhigh")] == [4096] * 3


def test_targets_without_much_reasoning_keep_700():
    assert [_max_output_tokens(e) for e in (None, "minimal", "low")] == [700] * 3
