"""Tests for the OpenRouter sim backend (--sim-backend openrouter).

Covers the assumptions the swap relies on, without a GPU, a live server, or a
network call (the OpenAI client is stubbed):

  1. `make_openrouter_call_llm` honors the harness `call_llm` signature and
     returns (text, in_tok, out_tok).
  2. It pins temperature=0. The frozen sim is greedy on the local backends, and
     `trainer._record_contribution` (resume without re-vetoing), the batched-vs-
     per-prompt equivalence claim, and the sequential-vs-concurrent veto parity
     all hold only for a deterministic judge.
  3. It floors max_tokens, so a reasoning model's hidden tokens cannot crowd out
     the visible answer on the harness's 128-token veto calls.
  4. Empty content fails loudly rather than propagating "" into the pipeline.
  5. The cost tracker meters every call and its cap aborts the run.
  6. `SimulatedStakeholders.max_concurrency` clamps the Trainer's fan-out (which
     is tuned for a local vLLM server, default 128) without changing results,
     and leaves local backends (max_concurrency=None) untouched.
  7. `Trainer._rewrite_fn` routes to the injected backend only when
     `rewrite_call_llm` is set, and to the policy otherwise (so every existing
     perspshift arm is unchanged).

Run with the project interpreter:
  /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/tests/test_openrouter_sim.py
"""

import os
import sys
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "core"))

import openrouter_sim
from cost import CostTracker
from simulated_stakeholders import SimulatedStakeholders


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, content, finish_reason="stop", usage=(11, 7)):
        self.choices = [_FakeChoice(content, finish_reason)]
        self.usage = None if usage is None else _FakeUsage(*usage)


class _FakeCompletions:
    """Records every create() kwargs; replies from a scripted queue."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return r


class _FakeClient:
    def __init__(self, responses):
        self.chat = type("_Chat", (), {})()
        self.chat.completions = _FakeCompletions(responses)


def _patched_call_llm(responses, monkey_client, **kw):
    """Build a call_llm with the OpenAI constructor stubbed to return
    `monkey_client`. Restores the real symbol afterwards.
    """
    saved = openrouter_sim.OpenAI
    openrouter_sim.OpenAI = lambda **_: monkey_client
    try:
        return openrouter_sim.make_openrouter_call_llm(**kw)
    finally:
        openrouter_sim.OpenAI = saved


def test_signature_and_return():
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    client = _FakeClient([_FakeResponse("  ACCEPT: fine.  ", usage=(11, 7))])
    call_llm = _patched_call_llm(None, client, model_id="openai/gpt-5-mini")
    # Harness signature: client + model are passed and ignored.
    text, in_tok, out_tok = call_llm(None, "ignored-model", "prompt?", 128,
                                     system="sys")
    assert text == "ACCEPT: fine.", text
    assert (in_tok, out_tok) == (11, 7), (in_tok, out_tok)
    kw = client.chat.completions.calls[0]
    assert kw["model"] == "openai/gpt-5-mini", kw["model"]
    assert kw["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prompt?"},
    ], kw["messages"]
    print("ok: signature + return shape + system routing")


def test_temperature_pinned_and_max_tokens_floored():
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    client = _FakeClient([_FakeResponse("VETO")])
    call_llm = _patched_call_llm(None, client, model_id="m",
                                 max_tokens_floor=4000)
    call_llm(None, None, "p", 128)  # the harness's veto_max_tokens
    kw = client.chat.completions.calls[0]
    assert kw["temperature"] == 0.0, (
        f"frozen sim must be greedy; got temperature={kw.get('temperature')!r}")
    assert kw["max_tokens"] == 4000, (
        f"128 must be floored to 4000 or a reasoning model returns empty "
        f"content; got {kw['max_tokens']}")
    assert kw["extra_body"] == {"reasoning": {"effort": "minimal"}}, kw

    # A request ABOVE the floor is passed through unchanged.
    client2 = _FakeClient([_FakeResponse("x")])
    call_llm2 = _patched_call_llm(None, client2, model_id="m",
                                  max_tokens_floor=4000)
    call_llm2(None, None, "p", 8192)
    assert client2.chat.completions.calls[0]["max_tokens"] == 8192
    print("ok: temperature pinned to 0, max_tokens floored")


def test_reasoning_none_omits_field():
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    client = _FakeClient([_FakeResponse("x")])
    call_llm = _patched_call_llm(None, client, model_id="m", reasoning=None)
    call_llm(None, None, "p", 128)
    assert "extra_body" not in client.chat.completions.calls[0]
    print("ok: reasoning=None omits the field")


def test_empty_content_asserts():
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    client = _FakeClient([_FakeResponse("", finish_reason="length")])
    call_llm = _patched_call_llm(None, client, model_id="m")
    try:
        call_llm(None, None, "p", 128)
    except AssertionError as e:
        assert "empty content" in str(e), str(e)
        print("ok: empty content asserts instead of returning ''")
        return
    raise AssertionError("expected an AssertionError on empty content")


def test_missing_key_asserts():
    saved = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        openrouter_sim.make_openrouter_call_llm(model_id="m")
    except AssertionError as e:
        assert "OPENROUTER_API_KEY" in str(e)
        print("ok: missing OPENROUTER_API_KEY asserts at construction")
        return
    finally:
        if saved is not None:
            os.environ["OPENROUTER_API_KEY"] = saved
    raise AssertionError("expected an AssertionError for a missing API key")


def test_cost_tracker_meters_and_caps():
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    # 1M in @ $1/1M + 1M out @ $2/1M = $3 per call; cap at $5 => 2nd call trips.
    tracker = CostTracker(price_in=1.0, price_out=2.0, max_cost=5.0,
                          label="OpenRouter sim")
    client = _FakeClient([_FakeResponse("x", usage=(1_000_000, 1_000_000))])
    call_llm = _patched_call_llm(None, client, model_id="m",
                                 cost_tracker=tracker)
    call_llm(None, None, "p", 128)
    assert abs(tracker.spent() - 3.0) < 1e-9, tracker.spent()
    try:
        call_llm(None, None, "p", 128)
    except RuntimeError as e:
        assert "OpenRouter sim" in str(e), str(e)
        print("ok: cost tracked per call; cap aborts with the sim's label")
        return
    raise AssertionError("expected a RuntimeError once the cap was reached")


# ---- SimulatedStakeholders.max_concurrency -------------------------------

POST = "My roommate keeps borrowing my car without asking."
ENTRIES = [{"stakeholder": "the roommate", "concerns": "needs transport"},
           {"stakeholder": "the narrator", "concerns": "wants the car"}]
# Appears in exactly one of the two candidate responses below.
VETO_MARKER = "ZZ-ENDORSE-ZZ"
RESPONSES = [f"Just take the car. {VETO_MARKER}", "Talk to your roommate."]


def _sim_with_cache(max_concurrency, seen_widths):
    """A SimulatedStakeholders whose cache is pre-primed (so no pipeline calls
    happen) and whose veto backend records the pool width it was run at.
    """
    lock = threading.Lock()

    # Veto keys on a marker unique to ONE response (not the post, which appears
    # in every veto prompt), so the veto set is a strict subset and the
    # clamped-vs-unclamped comparison below can actually discriminate.
    def fake_call_llm(client, model, prompt, max_tokens, system=None):
        with lock:
            seen_widths.append(threading.active_count())
        return ("VETO: no." if VETO_MARKER in prompt else "ACCEPT: ok."), 1, 1

    sim = SimulatedStakeholders(
        call_llm=fake_call_llm,
        model_id="stub",
        veto_concurrency=1,
        max_concurrency=max_concurrency,
    )
    sim._cache[POST] = ([e["stakeholder"] for e in ENTRIES], ENTRIES)
    return sim


def test_max_concurrency_clamps_and_defaults():
    sim = _sim_with_cache(max_concurrency=4, seen_widths=[])
    assert sim._bounded(128) == 4, sim._bounded(128)
    assert sim._bounded(2) == 2, "a smaller caller request must not be raised"
    assert sim.veto_concurrency == 1

    # veto_concurrency is clamped at construction too.
    sim_hi = SimulatedStakeholders(call_llm=lambda *a, **k: ("ACCEPT", 1, 1),
                                   model_id="stub", veto_concurrency=64,
                                   max_concurrency=8)
    assert sim_hi.veto_concurrency == 8, sim_hi.veto_concurrency

    # Local backends pass None and keep the caller's number exactly.
    sim_local = _sim_with_cache(max_concurrency=None, seen_widths=[])
    assert sim_local._bounded(128) == 128, sim_local._bounded(128)
    print("ok: max_concurrency clamps every fan-out; None is a no-op")


def test_clamped_veto_matches_unclamped():
    """Clamping is a throughput knob, not a semantic one: the veto set must be
    identical with and without it."""
    a = _sim_with_cache(max_concurrency=1, seen_widths=[])
    b = _sim_with_cache(max_concurrency=None, seen_widths=[])
    out_a = a.veto_responses_batch([(POST, RESPONSES)], concurrency=128)
    out_b = b.veto_responses_batch([(POST, RESPONSES)], concurrency=128)
    assert out_a == out_b, (out_a, out_b)
    # Guard the guard: a test where everything (or nothing) is vetoed would pass
    # the equality check while discriminating nothing.
    assert out_a == [[RESPONSES[0]]], (
        f"expected exactly the marked response vetoed, got {out_a}")
    print(f"ok: clamped veto set == unclamped, strict subset ({out_a})")


# ---- Trainer._rewrite_fn -------------------------------------------------

def test_rewrite_fn_routing():
    """_rewrite_fn must use the injected backend only when one is given, and the
    policy otherwise (keeping every existing perspshift arm unchanged)."""
    from trainer import Trainer

    calls = []

    class _StubTrainer:
        rewrite_max_tokens = 1536
        _rewrite_fn = Trainer._rewrite_fn

        def _policy_generate_greedy(self, p, n):
            calls.append(("policy", p, n))
            return "third-person via policy"

    # Default: no injected backend -> the policy.
    t = _StubTrainer()
    t.rewrite_call_llm = None
    assert t._rewrite_fn()("rewrite this") == "third-person via policy"
    assert calls == [("policy", "rewrite this", 1536)], calls

    # Injected: the harness-signature closure, first element of its tuple.
    def fake_call_llm(client, model, prompt, max_tokens, system=None):
        calls.append(("sim", prompt, max_tokens))
        return "third-person via sim", 3, 4

    t2 = _StubTrainer()
    t2.rewrite_call_llm = fake_call_llm
    assert t2._rewrite_fn()("rewrite this") == "third-person via sim"
    assert calls[-1] == ("sim", "rewrite this", 1536), calls[-1]
    print("ok: rewrite routes to policy by default, to the sim when injected")


if __name__ == "__main__":
    test_signature_and_return()
    test_temperature_pinned_and_max_tokens_floored()
    test_reasoning_none_omits_field()
    test_empty_content_asserts()
    test_missing_key_asserts()
    test_cost_tracker_meters_and_caps()
    test_max_concurrency_clamps_and_defaults()
    test_clamped_veto_matches_unclamped()
    test_rewrite_fn_routing()
    print("\nall openrouter sim tests passed")
