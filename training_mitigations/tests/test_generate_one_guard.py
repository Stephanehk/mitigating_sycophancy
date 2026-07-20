"""generate_one must not let one corrupted generation abort the eval.

Multi-GPU `device_map=auto` decoding can rarely emit an out-of-range token id
(negative, or >= len(tokenizer)), which crashes `tokenizer.decode` with an
OverflowError. generate_one guards this: an out-of-range `new_ids` triggers ONE
regeneration; a clean retry proceeds, but a second out-of-range result raises
(systematic fault, not a flaky event).

This drives generate_one with a stub model/tokenizer and a monkeypatched
build_chat_input, asserting:
  clean first attempt          -> decoded text, no retry
  bad then clean               -> retry runs, decoded text from the retry
  bad twice                    -> AssertionError (fail fast, not silent mask)
  boundary ids (0, len-1)      -> valid; -1 and len are out of range

Run:
  /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/tests/test_generate_one_guard.py
"""

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), "mitigations"))

import eval_endorsement_heldout as E

VOCAB_N = 100  # stub tokenizer len; valid ids are [0, 100)
IN_LEN = 2     # stub prompt prefix length


class StubTokenizer:
    """len()==VOCAB_N; decode joins ids so the test can read them back."""
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return VOCAB_N

    def decode(self, ids, skip_special_tokens=True):
        # Mirror the real failure mode: an out-of-range id would raise here, so
        # generate_one must never hand us one.
        vals = [int(x) for x in ids]
        assert all(0 <= v < VOCAB_N for v in vals), \
            f"decode got out-of-range id(s): {vals}"
        return " ".join(str(v) for v in vals)


class StubModel:
    """generate() returns the next canned full sequence (prefix + new ids)."""

    def __init__(self, sequences):
        self._sequences = list(sequences)
        self._i = 0
        self.generate_calls = 0

    def parameters(self):
        # generate_one only reads next(...).device.
        yield torch.zeros(1)

    def generate(self, input_ids, **kwargs):
        self.generate_calls += 1
        seq = self._sequences[self._i]
        self._i += 1
        return seq


def _seq(new_ids):
    """A full [1, IN_LEN + len(new_ids)] output: zero prefix + the new ids."""
    prefix = [0] * IN_LEN
    return torch.tensor([prefix + list(new_ids)], dtype=torch.long)


def _patch_chat_input():
    """build_chat_input -> a fixed [1, IN_LEN] prompt tensor on cpu."""
    E.build_chat_input = lambda tok, prompt, system, device: \
        torch.zeros((1, IN_LEN), dtype=torch.long)


def _call(model):
    return E.generate_one(model, StubTokenizer(), "p", max_new_tokens=8,
                          temperature=1.0, top_p=0.95, greedy=True)


def test_clean_first_attempt_no_retry():
    _patch_chat_input()
    m = StubModel([_seq([5, 6, 7])])
    out = _call(m)
    assert out == "5 6 7", out
    assert m.generate_calls == 1, m.generate_calls
    print("clean first attempt -> no retry: OK")


def test_bad_then_clean_retries_once():
    _patch_chat_input()
    # First generation has a corrupted (negative) id; retry is clean.
    m = StubModel([_seq([5, -1, 7]), _seq([8, 9])])
    out = _call(m)
    assert out == "8 9", out
    assert m.generate_calls == 2, m.generate_calls
    print("bad then clean -> one retry, retry result used: OK")


def test_huge_id_also_triggers_retry():
    _patch_chat_input()
    # A >= len(tokenizer) id (the other OverflowError shape) also counts as bad.
    m = StubModel([_seq([5, VOCAB_N + 4, 7]), _seq([1, 2, 3])])
    out = _call(m)
    assert out == "1 2 3", out
    assert m.generate_calls == 2, m.generate_calls
    print("huge (>=len) id -> retry: OK")


def test_bad_twice_raises():
    _patch_chat_input()
    m = StubModel([_seq([5, -1, 7]), _seq([-3, 9])])
    try:
        _call(m)
    except AssertionError as e:
        assert "both attempts" in str(e), e
        assert m.generate_calls == 2, m.generate_calls
        print("bad twice -> raises (fail fast, no mask): OK")
    else:
        raise SystemExit("generate_one did not raise on a double corruption")


def test_boundary_ids_are_valid():
    _patch_chat_input()
    # 0 and len-1 are valid and decodable; must NOT be treated as corrupted.
    m = StubModel([_seq([0, VOCAB_N - 1])])
    out = _call(m)
    assert out == f"0 {VOCAB_N - 1}", out
    assert m.generate_calls == 1, m.generate_calls
    print("boundary ids 0 and len-1 valid -> no retry: OK")


if __name__ == "__main__":
    test_clean_first_attempt_no_retry()
    test_bad_then_clean_retries_once()
    test_huge_id_also_triggers_retry()
    test_bad_twice_raises()
    test_boundary_ids_are_valid()
    print("\nall generate_one guard tests passed")
