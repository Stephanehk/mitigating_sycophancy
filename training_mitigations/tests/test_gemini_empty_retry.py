"""Unit tests for the empty-content retry path in gemini_sim.make_gemini_call_llm.

Motivation: a Gemini call can return "successfully" (no exception) with an EMPTY
body and a NON-block finish_reason (observed: None, and MAX_TOKENS). That is
neither a raised transport fault (the transient-retry path) nor a deterministic
safety refusal (a _BLOCK_REASONS finish, raised as GeminiBlockedError). It used
to trip a bare assert and kill a multi-hour training run over one flaky blank.
These tests pin the fix: such a reply is retried on the same schedule, and only
after the LAST attempt does it raise — while a genuine block is still raised
immediately, never retried.

Everything is stubbed: genai.Client is replaced with a fake whose
`models.generate_content` replays a scripted sequence, and time.sleep is a
counter so the backoff schedule costs nothing. No network, no GPU.

Run with the project interpreter:
  /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/tests/test_gemini_empty_retry.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "core"))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

import gemini_sim
from sim_errors import SimRefusedError


class _FR:
    """A stand-in for a finish_reason enum; the code reads `.name`."""
    def __init__(self, name):
        self.name = name


class _Cand:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self):
        self.prompt_token_count = 5
        self.candidates_token_count = 7
        self.thoughts_token_count = 0


def _resp(text, finish_name):
    """A response with one candidate whose finish_reason.name == finish_name.

    finish_name=None models the observed literal finish_reason=None (the code
    reads getattr(finish, "name", str(finish)) -> "None").
    """
    r = type("Resp", (), {})()
    r.text = text
    r.candidates = [_Cand(None if finish_name is None else _FR(finish_name))]
    r.usage_metadata = _Usage()
    return r


class _FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate_content(self, model, contents, config):
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, script):
        self.models = _FakeModels(script)


def _make(script, **kw):
    """Build a call_llm bound to a fake client replaying `script`.

    Patches genai.Client (constructed inside make_gemini_call_llm) and
    time.sleep (no real backoff). Returns (call_llm, fake_client, sleeps).
    """
    holder = {}
    sleeps = {"n": 0}

    def _fake_client_ctor(api_key):
        holder["client"] = _FakeClient(script)
        return holder["client"]

    gemini_sim.genai.Client = _fake_client_ctor
    gemini_sim.time.sleep = lambda s: sleeps.__setitem__("n", sleeps["n"] + 1)

    fn = gemini_sim.make_gemini_call_llm("gemini-3.1-flash-lite", **kw)
    return fn, holder["client"], sleeps


def test_empty_none_retries_then_succeeds():
    fn, client, sleeps = _make(
        [_resp("", None), _resp("", None), _resp("hello", "STOP")],
        max_retries=5)
    text, i_tok, o_tok = fn(None, None, "prompt", 64)
    assert text == "hello", text
    assert (i_tok, o_tok) == (5, 7), (i_tok, o_tok)
    assert client.models.calls == 3, client.models.calls
    assert sleeps["n"] == 2, sleeps  # one backoff before each of the 2 retries
    print("test_empty_none_retries_then_succeeds: OK")


def test_empty_maxtokens_is_retried():
    fn, client, _ = _make([_resp("", "MAX_TOKENS"), _resp("ok", "STOP")],
                          max_retries=5)
    text, _, _ = fn(None, None, "prompt", 64)
    assert text == "ok", text
    assert client.models.calls == 2, client.models.calls
    print("test_empty_maxtokens_is_retried: OK")


def test_empty_exhausts_retries_then_asserts():
    fn, client, sleeps = _make([_resp("", None)] * 3, max_retries=3)
    try:
        fn(None, None, "prompt", 64)
    except AssertionError as e:
        assert "empty content" in str(e), e
        assert "final of 3 retries" in str(e), e
        assert client.models.calls == 3, client.models.calls
        assert sleeps["n"] == 2, sleeps  # backoff before attempts 2 and 3, none after the last
        print("test_empty_exhausts_retries_then_asserts: OK")
        return
    raise SystemExit("expected AssertionError after exhausting retries")


def test_block_raises_immediately_without_retry():
    seen = {}
    fn, client, sleeps = _make([_resp("", "SAFETY"), _resp("late", "STOP")],
                               max_retries=5,
                               on_blocked=lambda p, r: seen.update(reason=r))
    try:
        fn(None, None, "an AITA post about minors", 64)
    except SimRefusedError as e:
        assert "SAFETY" in str(e), e
        assert client.models.calls == 1, "a block must NOT be retried"
        assert sleeps["n"] == 0, sleeps
        assert seen.get("reason") == "SAFETY", seen
        print("test_block_raises_immediately_without_retry: OK")
        return
    raise SystemExit("expected SimRefusedError on a block finish_reason")


def test_nonempty_success_short_circuits():
    fn, client, _ = _make([_resp("done", "STOP")], max_retries=5)
    text, _, _ = fn(None, None, "prompt", 64)
    assert text == "done"
    assert client.models.calls == 1
    print("test_nonempty_success_short_circuits: OK")


if __name__ == "__main__":
    test_empty_none_retries_then_succeeds()
    test_empty_maxtokens_is_retried()
    test_empty_exhausts_retries_then_asserts()
    test_block_raises_immediately_without_retry()
    test_nonempty_success_short_circuits()
    print("\nall gemini empty-retry tests passed")
