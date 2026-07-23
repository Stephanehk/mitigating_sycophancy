"""Tests for the two new mixed-sampling methods:

  1. future_user_steer (core/trainer.py): the n_source_critical slots become the
     future-self voice — tagged "future_user", zero "critical" — and the
     mutual-exclusion / budget asserts fire.
  2. addressee_filter (core/simulated_stakeholders.py): a candidate the filter
     labels 1 (message-critique) is removed from BOTH the vetoed and non-vetoed
     sets (unioned into _last_excluded AND pruned from the vetoed list), a label-0
     candidate is untouched, and the union does not clobber the bidirectional
     path's unparsable-vote exclusions.

Runs without a real model or vLLM server: every LLM primitive is stubbed with
deterministic Python. Run:
    /projects/m000254/envs/llm126/bin/python \
        training_mitigations/tests/test_new_methods.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, _HERE)


def _raises_assertion(fn):
    """True iff calling fn() raises AssertionError (repo style: no pytest dep)."""
    try:
        fn()
    except AssertionError:
        return True
    return False


def _make_trainer(**kw):
    """A Trainer with the mixed-sampling defaults this suite needs. model/tokenizer
    are None (construction is model-free, as in test_batched_pipeline)."""
    from trainer import Trainer
    base = dict(sim_stake=None, model=None, tokenizer=None, n_samples=10,
                temperature=1.0, mixed_sampling=True,
                n_source_model=2, n_source_critical=2)
    base.update(kw)
    return Trainer(**base)


def test_future_user_source_tags():
    from trainer import FUTURE_USER_RESPONSE_PROMPT
    tr = _make_trainer(future_user_steer=True)
    calls = []

    def fake_gen(gen_prompt, n, use_base=False):
        calls.append((gen_prompt, n, use_base))
        return ["r%d_%d" % (len(calls), i) for i in range(n)]
    tr._gen = fake_gen

    entries = [{"stakeholder": s, "concerns": ["c"]} for s in ("A", "B", "C")]
    responses, sources = tr._sample_responses_mixed("POST", entries)

    assert len(responses) == 10 and len(sources) == 10, (responses, sources)
    assert sources.count("model") == 2, sources
    assert sources.count("future_user") == 2, sources
    assert sources.count("critical") == 0, sources
    assert sum(s.startswith("stakeholder:") for s in sources) == 6, sources
    # The 2 future-user candidates were generated ON-POLICY (use_base False) from
    # the FUTURE_USER prompt, NOT from "Be critical."
    fu_prompt = FUTURE_USER_RESPONSE_PROMPT.format(post="POST")
    assert any(gp == fu_prompt and not ub for gp, n, ub in calls), calls
    assert all("Be critical." not in gp for gp, n, ub in calls), calls
    print("ok: future_user source tags (2 model / 2 future_user / 6 stakeholder)")


def test_future_user_asserts():
    # Mutually exclusive with perspshift (both consume the critical budget):
    assert _raises_assertion(
        lambda: _make_trainer(future_user_steer=True,
                              perspective_shift_steer=True))
    # Needs a non-zero critical budget:
    assert _raises_assertion(
        lambda: _make_trainer(future_user_steer=True, n_source_critical=0))
    # Needs mixed_sampling:
    assert _raises_assertion(
        lambda: _make_trainer(future_user_steer=True, mixed_sampling=False))
    # Sanity: the valid config does NOT raise.
    _make_trainer(future_user_steer=True)
    print("ok: future_user asserts (exclusivity, budget, mixed_sampling)")


def _make_sim(**kw):
    from simulated_stakeholders import SimulatedStakeholders
    base = dict(call_llm=lambda *a, **k: ("", 0, 0), model_id="stub",
                use_weak_model=True, veto_concurrency=4)
    base.update(kw)
    sim = SimulatedStakeholders(**base)
    # Neutralize priming so the unit test never runs the real identify pipeline.
    sim.prime_stakeholder_cache = lambda prompts, concurrency: None
    return sim


def test_addressee_filter_single_stage():
    sim = _make_sim(addressee_filter=True)
    # Single-stage veto: a stakeholder vetoes iff its name is in the response.
    sim._stakeholder_vetos = lambda prompt, response, entry: (
        entry["stakeholder"] in response, "")
    # Addressee filter: flag any response containing "CRITIQUE" (label 1). The
    # classifier now returns (is_critique, parse_ok).
    sim._classify_addressee = lambda prompt, response: ("CRITIQUE" in response, True)
    sim._cache["p"] = (["S1"], [{"stakeholder": "S1", "concerns": []}])

    responses = ["clean", "S1 vetoed", "CRITIQUE form", "S1 CRITIQUE both"]
    out = sim.veto_responses_batch([("p", responses)], concurrency=4)

    # CHANGE 1: a filtered-but-vetoed response STAYS a rejected exemplar. So
    # "S1 CRITIQUE both" (vetoed by S1 AND filtered) remains in the vetoed list,
    # while ALSO being in _last_excluded (dropped from the chosen set).
    assert out[0] == ["S1 vetoed", "S1 CRITIQUE both"], out
    assert set(sim._last_excluded[0]) == {"CRITIQUE form", "S1 CRITIQUE both"}, \
        sim._last_excluded
    assert "S1 CRITIQUE both" in out[0] and "S1 CRITIQUE both" in sim._last_excluded[0]
    # A label-0 candidate is neither vetoed nor excluded (so it stays choosable).
    assert "clean" not in out[0] and "clean" not in sim._last_excluded[0]
    print("ok: addressee filter single-stage (filtered+vetoed stays rejected, "
          "dropped from chosen)")


def test_addressee_filter_unions_with_bidirectional():
    from simulated_stakeholders import ENDORSE_CLASSIFY_PROMPT_P5, VETO_PROMPT_P6
    sim = _make_sim(addressee_filter=True,
                    endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPT_P5,
                    veto_prompt=VETO_PROMPT_P6, bidirectional=True)
    sim._classify_endorsement = lambda prompt, response: (True, "j")

    def bidir(prompt, response, entry, endorses, just):
        # An unparsable vote (None) on "UNPARSE" -> bidirectional excludes it.
        if "UNPARSE" in response:
            return None, "x"
        return False, "ACCEPT"
    sim._bidirectional_veto = bidir
    sim._classify_addressee = lambda prompt, response: ("CRITIQUE" in response, True)
    sim._cache["p"] = (["S1"], [{"stakeholder": "S1", "concerns": []}])

    responses = ["plain", "UNPARSE me", "CRITIQUE form"]
    out = sim.veto_responses_batch([("p", responses)], concurrency=4)

    # Nothing cleanly vetoed; the bidirectional path excludes "UNPARSE me", and
    # the addressee filter's "CRITIQUE form" is UNIONED in (not clobbered). This
    # also exercises the FUSED Stage-1 wave (endorse + addressee together).
    assert out[0] == [], out
    assert set(sim._last_excluded[0]) == {"UNPARSE me", "CRITIQUE form"}, \
        sim._last_excluded
    print("ok: addressee filter unions with bidirectional exclusions (no clobber)")


def test_addressee_unparseable_allows_through():
    # CHANGE 2: an unparseable label ALLOWS the candidate through (fail open) and
    # is COUNTED (not asserted). Drive the REAL _classify_addressee via a veto
    # backend that returns a token-free reply.
    sim = _make_sim(addressee_filter=True)
    sim.veto_call_llm = lambda client, model, prompt, max_tokens: ("banana", 0, 0)
    sim._stakeholder_vetos = lambda prompt, response, entry: (False, "")
    sim._cache["p"] = (["S1"], [{"stakeholder": "S1", "concerns": []}])

    out = sim.veto_responses_batch([("p", ["a", "b"])], concurrency=2)

    # Unparseable -> neither filtered nor vetoed; both candidates survive.
    assert out[0] == [] and sim._last_excluded[0] == [], (out, sim._last_excluded)
    # ...and the unparseable labels were counted, not asserted.
    assert sim._addressee_total == 2 and sim._addressee_unparsed == 2, (
        sim._addressee_total, sim._addressee_unparsed)
    print("ok: addressee unparseable allows through + counters increment")


if __name__ == "__main__":
    test_future_user_source_tags()
    test_future_user_asserts()
    test_addressee_filter_single_stage()
    test_addressee_filter_unions_with_bidirectional()
    test_addressee_unparseable_allows_through()
    print("\nALL PASS")
