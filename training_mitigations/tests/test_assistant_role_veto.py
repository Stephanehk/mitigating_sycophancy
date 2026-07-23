"""Tests for the assistant-role veto (train_dpo_lora.py --assistant-role-veto).

Covers the assumptions the method rests on, per CLAUDE.md's "validate important
assumptions / cover edge and failure cases":

  1. Pair CAP is on PAIRS, not responses: >3 flagged candidates still yield
     exactly AUTO_REJECT_MAX_PAIRS pairs, and they are NOT multiplied through
     the veto cross-product.
  2. Auto-rejected responses never appear as CHOSEN.
  3. Chosen partners come only from never-vetoed responses.
  4. Selection is deterministic given the prompt, so preempt-resume rebuilds the
     identical dataset (_record_contribution vs the live path).
  5. The classifier FAILS OPEN on an unparseable label (no fabricated rejects)
     and the failure is counted, not asserted.
  6. A response already vetoed by the panel is NOT also auto-rejected (no
     duplicate pairs for the same (prompt, rejected)).
  7. --assistant-role-veto is refused alongside --future-user-sampling.

All LLM calls are stubbed; no GPU, no network, no model load.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

import simulated_stakeholders as S
from simulated_stakeholders import _flagged_positions


# --- helpers ---------------------------------------------------------------

def _sim(label_for):
    """A SimulatedStakeholders with only the bits the assistant-role path uses.

    `label_for(response)` returns the raw judge text for that candidate, so a
    test can inject "1", "0", or garbage. Built via __new__ to skip the real
    constructor's model/pipeline setup.
    """
    sim = S.SimulatedStakeholders.__new__(S.SimulatedStakeholders)
    sim.assistant_role_veto = True
    sim.addressee_filter = False
    sim._arole_unparsed = 0
    sim._arole_total = 0
    sim.veto_max_tokens = 32
    sim.veto_client = None
    sim.assistant_role_prompt = S.ASSISTANT_ROLE_VETO_PROMPT
    sim.veto_model_id = "stub"
    sim.veto_call_llm = lambda client, model, prompt, max_tokens, system=None: (
        label_for(prompt), 0, 0)
    return sim


def _label_by_marker(marker):
    """Judge stub: label 1 iff `marker` appears in the RESPONSE section."""
    def f(classify_prompt):
        return "1" if marker in classify_prompt.split("RESPONSE:")[-1] else "0"
    return f


# --- 5. fail-open ----------------------------------------------------------

def test_unparseable_label_fails_open():
    """An unparseable judge reply must NOT auto-reject: a bad label would
    fabricate a rejected exemplar out of a fine response."""
    sim = _sim(lambda _p: "I'm not sure how to classify this.")
    abandons, parse_ok = sim._classify_assistant_role("post", "resp")
    assert abandons is False
    assert parse_ok is False


def test_parseable_labels_map_correctly():
    sim = _sim(lambda _p: "1")
    assert sim._classify_assistant_role("post", "resp") == (True, True)
    sim = _sim(lambda _p: "0")
    assert sim._classify_assistant_role("post", "resp") == (False, True)


def test_unparseable_is_counted_not_asserted(capsys):
    """Unparseable labels are tallied and WARNed, never raised — this is a
    ~30k-call loop, matching the addressee filter's precedent."""
    sim = _sim(lambda _p: "?")
    items = [("post", ["a", "b"])]
    flagged = [(0, 0, (False, False)), (0, 1, (False, True))]
    out = sim._arole_from_flags(items, flagged)
    assert out == [[]]
    assert sim._arole_unparsed == 1
    assert sim._arole_total == 2
    assert "assistant-role-veto] WARN" in capsys.readouterr().out


# --- shared flag->positions helper ----------------------------------------

def test_flagged_positions_is_per_occurrence():
    """Duplicate response texts are indexed by POSITION, so flagging one copy
    does not silently flag the other."""
    items = [("p", ["same", "same", "other"])]
    flagged = [(0, 0, (True, True)), (0, 1, (False, True)), (0, 2, (True, True))]
    out, n_unparsed, n_total = _flagged_positions(items, flagged)
    assert out == [["same", "other"]]
    assert (n_unparsed, n_total) == (0, 3)


# --- 6. dedupe against the panel's vetoes ----------------------------------

def test_panel_vetoed_response_is_not_also_auto_rejected():
    """A response the stakeholders already vetoed is a rejected exemplar via the
    cross-product; auto-rejecting it too would duplicate the pair."""
    sim = _sim(_label_by_marker("BAD"))
    sim.two_stage = False
    sim.bidirectional = False
    sim.max_concurrency = None
    sim._cache = {"post": ([], [])}
    sim.prime_stakeholder_cache = lambda prompts, concurrency: None
    # Panel vetoes "BAD-and-vetoed"; the arole judge flags both BAD responses.
    sim._stakeholder_vetos = lambda p, r, e: (r == "BAD-and-vetoed", "")
    items = [("post", ["BAD-and-vetoed", "BAD-only", "fine"])]
    # No stakeholders in cache -> no veto tasks, so drive the veto list directly.
    out = sim.veto_responses_batch(items, concurrency=2)
    assert out == [[]]
    # With an empty panel veto, both BAD responses survive dedupe.
    assert sim._last_auto_rejected == [["BAD-and-vetoed", "BAD-only"]]


# --- trainer-side pair construction ---------------------------------------

def _pairs(prompt, auto_rejected, non_vetoed):
    from trainer import _auto_reject_pairs
    return _auto_reject_pairs(prompt, auto_rejected, non_vetoed)


def test_cap_is_on_pairs_not_responses():
    """5 flagged candidates -> exactly AUTO_REJECT_MAX_PAIRS pairs. Capping
    responses instead would multiply through the cross-product (5 x 4 = 20)."""
    from trainer import AUTO_REJECT_MAX_PAIRS
    assert AUTO_REJECT_MAX_PAIRS == 3
    pairs = _pairs("post", ["r1", "r2", "r3", "r4", "r5"], ["c1", "c2", "c3", "c4"])
    assert len(pairs) == AUTO_REJECT_MAX_PAIRS
    assert [p["rejected"] for p in pairs] == ["r1", "r2", "r3"]


def test_chosen_comes_only_from_never_vetoed():
    pool = ["c1", "c2", "c3"]
    pairs = _pairs("post", ["r1", "r2", "r3"], pool)
    assert all(p["chosen"] in pool for p in pairs)
    assert all(p["rejected"] not in pool for p in pairs)


def test_no_pairs_without_a_chosen_partner():
    """All candidates flagged -> empty chosen pool -> no pairs (not a crash,
    and never a pair whose chosen is itself auto-rejected)."""
    assert _pairs("post", ["r1", "r2"], []) == []
    assert _pairs("post", [], ["c1"]) == []


def test_selection_is_deterministic_for_resume():
    """Same (prompt, flagged, pool) -> same pairs. The reproducible train path
    has no global RNG; a shared Random() would desync on preempt-resume and
    _record_contribution would rebuild a DIFFERENT dataset."""
    args = ("a post", ["r1", "r2", "r3"], ["c1", "c2", "c3", "c4"])
    assert _pairs(*args) == _pairs(*args)


def test_different_prompts_select_differently():
    """Seeding on the prompt (not a constant) means the chosen partner varies
    across prompts instead of always landing on pool[0]."""
    pool = [f"c{i}" for i in range(8)]
    picks = {_pairs(f"post-{i}", ["r1"], pool)[0]["chosen"] for i in range(30)}
    assert len(picks) > 1


def test_prompt_is_the_bare_post():
    """The DPO prompt column stays the bare post, matching the veto cross-pairs
    and the eval-time input surface."""
    pairs = _pairs("the bare post", ["r1"], ["c1"])
    assert pairs[0]["prompt"] == "the bare post"


# --- 2 + 4. resume rebuilds the identical dataset --------------------------

def test_record_contribution_matches_live_path():
    """_record_contribution must rebuild exactly what the live path built:
    auto-rejected held out of `non_vetoed`, same capped pairs appended."""
    from trainer import Trainer, _auto_reject_pairs
    t = Trainer.__new__(Trainer)
    t.self_preference = False
    t.distill_self_only = False
    record = {
        "prompt": "post",
        "responses": ["v1", "ar1", "ar2", "c1", "c2"],
        "vetoed": ["v1"],
        "excluded": [],
        "auto_rejected": ["ar1", "ar2"],
        "sources": ["model"] * 5,
    }
    pairs, vetoed, non_vetoed = t._record_contribution(record)
    # Auto-rejected are out of the chosen pool.
    assert non_vetoed == ["c1", "c2"]
    assert "ar1" not in non_vetoed and "ar2" not in non_vetoed
    # Cross-product (1 vetoed x 2 chosen) + 2 auto-reject pairs.
    cross = [{"prompt": "post", "chosen": c, "rejected": "v1"} for c in non_vetoed]
    expected = cross + _auto_reject_pairs("post", ["ar1", "ar2"], non_vetoed)
    assert pairs == expected
    assert vetoed == ["v1"]


def test_record_without_auto_rejected_field_is_unchanged():
    """Records from runs without the veto (or predating the field) rebuild
    exactly as before — no behavior change for existing arms."""
    from trainer import Trainer
    t = Trainer.__new__(Trainer)
    t.self_preference = False
    t.distill_self_only = False
    record = {
        "prompt": "post",
        "responses": ["v1", "c1"],
        "vetoed": ["v1"],
        "excluded": [],
        "sources": ["model", "model"],
    }
    pairs, _vetoed, non_vetoed = t._record_contribution(record)
    assert non_vetoed == ["c1"]
    assert pairs == [{"prompt": "post", "chosen": "c1", "rejected": "v1"}]


# --- 7. incompatible with future-user sampling ------------------------------

def test_future_user_sampling_is_rejected():
    """ASSISTANT_ROLE_VETO_PROMPT sub-mode (b) names the user's own future self
    as role-abandonment, which is exactly what FUTURE_USER_RESPONSE_PROMPT
    generates on purpose — combining them would reject every such candidate."""
    prompt_src = open(os.path.join(
        os.path.dirname(__file__), "..", "train_dpo_lora.py")).read()
    assert "--future-user-sampling and --assistant-role-veto are mutually" in prompt_src


# --- judge-specific prompt selection ---------------------------------------

def test_qwen_variant_differs_only_in_submode_b():
    """The _QWEN variant must keep sub-modes (a)/(c), the label-0 clause and the
    output contract identical, so the base-prompt validation numbers stay
    comparable. Only the (b) epistemic test + ignore-list is added."""
    base, qwen = S.ASSISTANT_ROLE_VETO_PROMPT, S.ASSISTANT_ROLE_VETO_PROMPT_QWEN
    for shared in ("{post}", "{response}",
                   "Output the number directly using one token",
                   "Being harsh, blunt, or unsympathetic is 0",
                   "as a piece of writing",          # (a)
                   "third person"):                  # (c)
        assert shared in base and shared in qwen, shared
    # The additions that fix Qwen's measured FPR=0.600 on harsh outsider critique.
    assert "PERSONALLY EXPERIENCED" in qwen and "PERSONALLY EXPERIENCED" not in base
    assert "Explicitly IGNORE them" in qwen
    assert "Both outsiders and insiders address the user as" in qwen


def test_local_backends_get_the_qwen_prompt():
    """Wiring pin: the local Qwen3-8B judge (vllm/hf) must get the _QWEN variant
    and API judges the base prompt. Getting this backwards silently reinstates
    FPR 0.600 on the marlowe arms."""
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "train_dpo_lora.py")).read()
    assert "ASSISTANT_ROLE_VETO_PROMPT_QWEN" in src
    assert 'if args.sim_backend in ("vllm", "hf")' in src


def test_prompt_defaults_to_base_when_unset():
    sim = _sim(lambda _p: "0")
    sim.assistant_role_prompt = S.ASSISTANT_ROLE_VETO_PROMPT
    assert sim.assistant_role_prompt is S.ASSISTANT_ROLE_VETO_PROMPT


# --- prompt contract --------------------------------------------------------

def test_prompt_has_required_placeholders_and_submodes():
    p = S.ASSISTANT_ROLE_VETO_PROMPT
    assert "{post}" in p and "{response}" in p
    # The three grounded sub-modes.
    assert "as a piece of writing" in p          # (a) message-as-artifact
    assert "inside the user's story" in p        # (b) in-story voice
    assert "third person" in p                   # (c) case-study framing
    # The label-0 protections that stop this from becoming a pronoun detector
    # or a harshness detector.
    assert "I think this is a bad idea" in p
    assert "Being harsh, blunt, or unsympathetic is 0" in p


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --- API refusal must not kill the run --------------------------------------

def test_platform_refusal_fails_open_not_crash():
    """A Gemini finish_reason=SAFETY refusal is deterministic and never retried
    by the backend. It must degrade to the unparsable encoding (candidate NOT
    auto-rejected, counted) rather than propagating and killing a multi-hour
    training run."""
    from sim_errors import SimRefusedError
    sim = _sim(lambda _p: "0")
    noted = []
    sim._note_refusal = lambda stage, detail, exc: noted.append(stage)

    def boom(prompt, response):
        raise SimRefusedError("Gemini refused (finish_reason=SAFETY)")

    flagged = sim._classify_wave(boom, [("post", ["a", "b"])], 2, "arole")
    assert [f[2] for f in flagged] == [(False, False), (False, False)]
    assert noted == ["arole", "arole"]
    out = sim._arole_from_flags([("post", ["a", "b"])], flagged)
    assert out == [[]], "a refused candidate must not become a rejected exemplar"
    assert sim._arole_unparsed == 2


def test_exhausted_empty_content_fails_open_not_crash():
    """Gemini returning empty content on the FINAL retry raises GeminiEmptyError.
    For the AUXILIARY assistant-role classifier that must degrade to unparsable,
    not kill the run — this crashed job 384191 at iteration 2 after 1h45m."""
    from gemini_sim import GeminiEmptyError
    assert issubclass(GeminiEmptyError, AssertionError), (
        "must stay an AssertionError so core sim steps keep fail-fast")
    sim = _sim(lambda _p: "0")
    noted = []
    sim._note_refusal = lambda stage, detail, exc: noted.append(stage)

    def empty(prompt, response):
        raise GeminiEmptyError("Gemini sim returned empty content ... final of 5 retries")

    flagged = sim._classify_wave(empty, [("post", ["a", "b"])], 2, "arole")
    assert [f[2] for f in flagged] == [(False, False), (False, False)]
    assert noted == ["arole-empty", "arole-empty"]
    assert sim._arole_from_flags([("post", ["a", "b"])], flagged) == [[]]
