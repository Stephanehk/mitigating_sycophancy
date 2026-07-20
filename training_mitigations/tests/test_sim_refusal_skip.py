"""A backend refusal must SKIP a step, never kill the run — and never quietly
promote a candidate to CHOSEN.

Gemini's content filter returns PROHIBITED_CONTENT deterministically (retrying
is useless), so `gemini_sim` raises `SimRefusedError` and the pipeline applies a
per-stage policy. This test drives a refusal at each stage with a stub call_llm
and asserts the right thing is discarded:

  identify/filter refused -> empty panel  => 0 vetoes => no pairs => prompt skipped
  concerns refused        -> DROP that stakeholder, keep the prompt's others
  stage-1 refused         -> EXCLUDE that response (stance unknown)
  stage-2 refused         -> drop only THAT VOTER'S vote; others still decide
  stage-2 all refused     -> EXCLUDE the response (no evidence either way)

The safety property throughout: a refusal must never make a response CHOSEN,
because the trainer derives chosen = responses - vetoed - excluded.

Run:
  /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/tests/test_sim_refusal_skip.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), "mitigations"))

from sim_errors import SimRefusedError
from simulated_stakeholders import SimulatedStakeholders, VETO_PROMPTS, ENDORSE_CLASSIFY_PROMPTS

POST = "I told my sister she was wrong."
RESPS = ["you were right", "you were wrong"]

# Markers unique to ONE stage's prompt each; test_markers_are_unique asserts it.
# NB "CONCERNS:" is NOT usable -- VETO_PROMPT_P6 embeds it too.
M_IDENTIFY = "List between"
M_FILTER = "previously identified"
M_CONCERNS = "From your perspective as"
M_STAGE1 = "Does the response endorse the narrator's own action"
STAKEHOLDERS = "Sister\nMother"
CONCERNS_REPLY = 'CONCERNS:\n- (action: "told her") you dismissed me'


def route(prompt, veto_reply="ACCEPT"):
    """Answer any pipeline prompt with a parser-valid reply for its stage."""
    if M_IDENTIFY in prompt or M_FILTER in prompt:
        return STAKEHOLDERS, 1, 1
    if M_CONCERNS in prompt:
        return CONCERNS_REPLY, 1, 1
    if M_STAGE1 in prompt:
        return "YES because it affirms", 1, 1
    return veto_reply, 1, 1


def make_sim(refuse_on, veto_reply="ACCEPT"):
    """Build a sim whose stub backend raises SimRefusedError for any prompt
    containing one of `refuse_on`. Replies are shaped to satisfy each stage's
    parser so only the refusal under test differs from a clean run."""
    def call_llm(client, model, prompt, max_tokens, system=None):
        for marker in refuse_on:
            if marker in prompt:
                raise SimRefusedError(f"stub refusal on {marker!r}")
        return route(prompt, veto_reply)
    return SimulatedStakeholders(
        call_llm=call_llm, model_id="stub",
        veto_prompt=VETO_PROMPTS["P6"],
        endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPTS["P5"],
        bidirectional=True)


def test_identify_refused_gives_empty_panel():
    sim = make_sim(["List between"])
    sim.prime_stakeholder_cache([POST], concurrency=1)
    cands, entries = sim._cache[POST]
    assert entries == [], f"expected empty panel, got {entries}"
    vetoed = sim.veto_responses_batch([(POST, RESPS)], concurrency=1)
    assert vetoed == [[]], f"empty panel must veto nothing, got {vetoed}"
    assert sim.refusal_summary() == {"identify/filter": 1}, sim.refusal_summary()
    print("identify refused -> empty panel, 0 vetoes, prompt skipped: OK")


def test_concerns_refused_drops_only_that_stakeholder():
    # Refuse concerns for Mother only (her name appears in her concerns prompt).
    def call_llm(client, model, prompt, max_tokens, system=None):
        if M_CONCERNS in prompt and "Mother" in prompt:
            raise SimRefusedError("stub refusal on Mother concerns")
        return route(prompt)
    sim = SimulatedStakeholders(call_llm=call_llm, model_id="stub",
                                veto_prompt=VETO_PROMPTS["P6"],
                                endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPTS["P5"],
                                bidirectional=True)
    sim.prime_stakeholder_cache([POST], concurrency=1)
    _c, entries = sim._cache[POST]
    names = [e["stakeholder"] for e in entries]
    assert names == ["Sister"], f"Mother should be dropped, got {names}"
    assert sim.refusal_summary() == {"concerns": 1}, sim.refusal_summary()
    print("concerns refused -> that stakeholder dropped, others kept: OK")


def test_stage1_refused_excludes_response():
    sim = make_sim([M_STAGE1])
    vetoed = sim.veto_responses_batch([(POST, RESPS)], concurrency=1)
    excluded = sim._last_excluded[0]
    assert vetoed == [[]], f"a refused-stance response must not be vetoed: {vetoed}"
    assert set(excluded) == set(RESPS), f"both responses must be excluded: {excluded}"
    assert sim.refusal_summary().get("stage1:endorse") == 2, sim.refusal_summary()
    print("stage-1 refused -> response EXCLUDED (never chosen): OK")


def test_stage2_partial_refusal_keeps_other_voters():
    """One voter refuses, another votes VETO -> the VETO still lands."""
    def call_llm(client, model, prompt, max_tokens, system=None):
        # Refuse ONLY Mother's stage-2 vote: a veto prompt is the one that is
        # neither identify/filter/concerns/stage-1.
        if (M_IDENTIFY not in prompt and M_FILTER not in prompt
                and M_CONCERNS not in prompt and M_STAGE1 not in prompt
                and "Mother" in prompt):
            raise SimRefusedError("stub refusal: Mother's vote")
        return route(prompt, veto_reply="VETO")   # Sister vetoes
    sim = SimulatedStakeholders(call_llm=call_llm, model_id="stub",
                                veto_prompt=VETO_PROMPTS["P6"],
                                endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPTS["P5"],
                                bidirectional=True)
    vetoed = sim.veto_responses_batch([(POST, RESPS)], concurrency=1)
    assert set(vetoed[0]) == set(RESPS), \
        f"Sister's VETO must survive Mother's refusal, got {vetoed}"
    assert sim._last_excluded[0] == [], \
        f"a partly-refused response must NOT be excluded: {sim._last_excluded[0]}"
    assert sim.refusal_summary().get("stage2:veto", 0) > 0, sim.refusal_summary()
    print("stage-2 partial refusal -> other voters still decide: OK")


def test_stage2_total_refusal_excludes_response():
    """Every voter refuses -> no evidence -> exclude, do NOT default to chosen."""
    def call_llm(client, model, prompt, max_tokens, system=None):
        if (M_IDENTIFY not in prompt and M_FILTER not in prompt
                and M_CONCERNS not in prompt and M_STAGE1 not in prompt):
            raise SimRefusedError("stub refusal: every vote")
        return route(prompt)
    sim = SimulatedStakeholders(call_llm=call_llm, model_id="stub",
                                veto_prompt=VETO_PROMPTS["P6"],
                                endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPTS["P5"],
                                bidirectional=True)
    vetoed = sim.veto_responses_batch([(POST, RESPS)], concurrency=1)
    assert vetoed == [[]], vetoed
    assert set(sim._last_excluded[0]) == set(RESPS), \
        f"all-refused responses must be EXCLUDED, not chosen: {sim._last_excluded[0]}"
    print("stage-2 total refusal -> response EXCLUDED (never chosen): OK")


def test_local_backend_unaffected():
    """No refusals => byte-identical behaviour to before (the Qwen path)."""
    sim = make_sim([], veto_reply="VETO")
    vetoed = sim.veto_responses_batch([(POST, RESPS)], concurrency=1)
    assert set(vetoed[0]) == set(RESPS), vetoed
    assert sim.refusal_summary() == {}, sim.refusal_summary()
    assert sim._last_excluded[0] == [], sim._last_excluded[0]
    print("no refusals -> local/Qwen path unchanged: OK")


def test_markers_are_unique():
    """Each stub marker must match its OWN stage's prompt and no other, or a
    test silently exercises the wrong stage and passes for the wrong reason."""
    from weak_pipeline import (CONCERNS_PROMPT_WEAK, STAKEHOLDER_PROMPT_WEAK,
                               FILTER_STAKEHOLDERS_PROMPT_WEAK)
    srcs = {"identify": STAKEHOLDER_PROMPT_WEAK, "filter": FILTER_STAKEHOLDERS_PROMPT_WEAK,
            "concerns": CONCERNS_PROMPT_WEAK, "stage1": ENDORSE_CLASSIFY_PROMPTS["P5"],
            "veto": VETO_PROMPTS["P6"]}
    for stage, mark in (("identify", M_IDENTIFY), ("filter", M_FILTER),
                        ("concerns", M_CONCERNS), ("stage1", M_STAGE1)):
        hits = [s for s, txt in srcs.items() if mark in txt]
        assert hits == [stage], f"marker {mark!r} matches {hits}, expected only {stage!r}"
    print("stub markers each match exactly one stage: OK")




def test_refused_prompt_is_skipped_by_mixed_sampling():
    """REGRESSION (killed a live launch): a refusal empties the panel, and mixed
    sampling draws candidates FROM the panel — so `_sample_responses_mixed`
    asserted and took the whole run down. A refused prompt must instead yield NO
    candidates (=> no pairs => skipped). An empty panel from any other cause must
    still assert, because that is a real bug."""
    import types
    from trainer import Trainer

    sim = make_sim(["List between"])                 # identify refused
    sim.prime_stakeholder_cache([POST], concurrency=1)
    assert sim._cache[POST][1] == [], "precondition: panel must be empty"
    assert POST in sim.refused_prompts, "refusal must be recorded per-prompt"

    # Minimal stand-in: only the fields _sample_responses_mixed reads before it
    # decides to skip. _gen would need a GPU, so reaching it is itself a failure.
    def boom(*a, **kw):
        raise AssertionError("generated candidates for a REFUSED prompt")
    fake = types.SimpleNamespace(
        n_source_model=2, n_source_critical=2, n_samples=10,
        sim_stake=sim, _gen=boom)

    responses, sources = Trainer._sample_responses_mixed(fake, POST, [])
    assert responses == [] and sources == [], (responses, sources)
    print("refused prompt -> mixed sampling skips it, no GPU work: OK")

    # Same shape, but NOT refused => the assert must still fire. All-stakeholder
    # budgets, and _gen is called with n=0 for the empty model slot (it is not
    # guarded), so return [] there and boom on any REAL generation.
    def gen0(_inp, n, **kw):
        if n == 0:
            return []
        raise AssertionError("generated candidates for a REFUSED prompt")
    sim2 = make_sim([])
    fake2 = types.SimpleNamespace(
        n_source_model=0, n_source_critical=0, n_samples=10,
        sim_stake=sim2, _gen=gen0,
        perspective_shift_steer=False, future_user_steer=False)
    try:
        Trainer._sample_responses_mixed(fake2, POST, [])
    except AssertionError as e:
        assert "no stakeholders" in str(e), e
        print("empty panel with NO refusal -> still asserts (real bug): OK")
    else:
        raise SystemExit("assert did not fire for a non-refusal empty panel")


if __name__ == "__main__":
    test_markers_are_unique()
    test_identify_refused_gives_empty_panel()
    test_concerns_refused_drops_only_that_stakeholder()
    test_stage1_refused_excludes_response()
    test_stage2_partial_refusal_keeps_other_voters()
    test_stage2_total_refusal_excludes_response()
    test_local_backend_unaffected()
    test_refused_prompt_is_skipped_by_mixed_sampling()
    print("\nall refusal-skip tests passed")
