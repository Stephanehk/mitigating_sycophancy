"""Tests for the seven arms' prompt construction.

The arm definitions are the study's independent variable, so the properties
that make the manipulation valid are asserted here rather than trusted:
  - assignment draws from three arms (RANDOMIZED_ARMS) that differ ONLY in how
    the response is produced: they share a flow, a user turn and a Stage-1
    system prompt. The four retired ones stay defined so old rows stay legal
  - every drawn arm SHOWS the review page, and they agree about it — a split
    would make the flow a second, visible factor
  - every arm receives the SAME shared system prompt ([ETHICS] only)
  - `syco` differs from that shared prompt by exactly one appended section
  - the removed sections ([ROLE], [CONTEXT], [GUIDING PRINCIPLES],
    [TONE AND FORMATTING - IMPORTANT], [TOPIC]) are gone from every arm
  - `ours` and `untrained` differ ONLY in which model is addressed
  - `ours_sandwich`'s Stage 1 is byte-identical to `ours`, so the pair isolates
    the framing pass with the substance held fixed
  - the two no-review arms differ from their twins ONLY in the flow — their
    generation plans are byte-identical, so the pair isolates the review step
  - the user turn is the participant's own writing, and no scenario text
  - no participant text ever lands in a system prompt
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prompts
import scenarios

# Both halves are the PARTICIPANT'S own words about their own incident: their
# screening description, then their side of the story. The scenario vignette is
# deliberately absent — see test_no_scenario_text_reaches_the_user_turn.
OWN_DESCRIPTION = "I spent a party talking to an acquaintance instead of my girlfriend."
PERSPECTIVE = "I felt distant from her and like we didn't have much to talk about."
USER_TEXT = prompts.build_user_turn(OWN_DESCRIPTION, PERSPECTIVE)

REMOVED_SECTIONS = ("[ROLE]", "[CONTEXT]", "[GUIDING PRINCIPLES]",
                    "[TONE AND FORMATTING - IMPORTANT]", "[TOPIC]")


# --- What random assignment draws from -------------------------------------
# Narrowed on 2026-08-05 to the two no-review conditions (it was the three model
# conditions from 2026-08-01, and all six before that). These pin the narrowing
# itself, since nothing else in Python would notice it being undone: the draw
# lives in SQL, and ARMS deliberately still lists all six.

def test_assignment_draws_the_three_response_arms():
    assert set(prompts.RANDOMIZED_ARMS) == {prompts.ARM_OURS,
                                            prompts.ARM_UNTRAINED,
                                            prompts.ARM_OURS_SANDWICH}
    assert set(prompts.RETIRED_ARMS) == set(prompts.ARMS) - set(prompts.RANDOMIZED_ARMS)


def test_the_retired_arms_are_still_defined_and_still_buildable():
    """Retired, not deleted. Rows already collected under them have to stay
    legal values, and keeping their plans buildable is what makes re-widening the
    draw a one-line change."""
    assert set(prompts.RETIRED_ARMS) == {prompts.ARM_SYCO,
                                         prompts.ARM_NEUTRAL,
                                         prompts.ARM_OURS_NOREVIEW,
                                         prompts.ARM_UNTRAINED_NOREVIEW}
    for arm in prompts.RETIRED_ARMS:
        assert arm in prompts.ARMS
        plan = prompts.build_generation_plan(arm, USER_TEXT)
        assert plan["arm"] == arm and plan["stages"]


def test_every_randomized_arm_shows_the_review_page():
    """The design rule as of 2026-08-07: every participant is shown the composed
    model input and may edit it, so response_flow answers True for all of them."""
    assert not set(prompts.RANDOMIZED_ARMS) & set(prompts.SKIP_REVIEW_ARMS)
    for arm in prompts.RANDOMIZED_ARMS:
        assert prompts.shows_review(arm) is True


def test_the_review_page_is_constant_across_the_draw():
    """The rule that outlives whichever way the constant points.

    A draw that mixed a review arm with a no-review arm would make the flow a
    second factor — and a VISIBLE one, so not something a covariate could take
    back out afterwards. This is the check that survives flipping the study back
    the other way; the test above is the one that says which way it points now.
    """
    shown = {prompts.shows_review(arm) for arm in prompts.RANDOMIZED_ARMS}
    assert len(shown) == 1, (
        "the drawable arms disagree about the review page: %r" % (
            {arm: prompts.shows_review(arm) for arm in prompts.RANDOMIZED_ARMS},))


def test_the_drawn_arms_differ_only_in_how_the_response_is_produced():
    """THE DESIGN OF THE THREE-WAY DRAW.

    Every drawn arm gets the SAME first user turn and the SAME Stage-1 system
    prompt, so the only thing that varies across cells is the production of the
    response: base model, trained model, or trained model plus a framing pass.
    A difference in the DVs cannot be attributed to what the model was asked, or
    to what the participant was shown before it answered.
    """
    plans = [prompts.build_generation_plan(arm, USER_TEXT)
             for arm in prompts.RANDOMIZED_ARMS]
    assert len({p["stages"][0]["system"] for p in plans}) == 1, (
        "the drawn arms no longer share a Stage-1 system prompt")
    assert len({p["stages"][0]["user"] for p in plans}) == 1, (
        "the drawn arms no longer share a user turn")
    for plan in plans:
        assert plan["stages"][0]["user"] == USER_TEXT
    signatures = [(tuple(s["adapter"] for s in p["stages"]),
                   tuple(s["system"] for s in p["stages"])) for p in plans]
    assert len(set(signatures)) == len(plans), "two drawable arms generate identically"


def test_the_sandwich_arm_is_ours_plus_a_base_model_framing_pass():
    """ours vs ours_sandwich must isolate the framing and nothing else.

    Stage 1 is taken FROM `ours`'s plan rather than restated, so this asserts an
    identity the builder cannot drift out of: change `ours` and the sandwich
    arm's first stage changes with it.
    """
    plan = prompts.build_generation_plan(prompts.ARM_OURS_SANDWICH, USER_TEXT)
    ours = prompts.build_generation_plan(prompts.ARM_OURS, USER_TEXT)
    assert [s["name"] for s in plan["stages"]] == ["generate", "sandwich"]
    assert plan["stages"][0] == ours["stages"][0], (
        "the sandwich arm's Stage 1 is no longer byte-identical to `ours`")
    framing = plan["stages"][1]
    # The BASE model frames; our adapter is trained not to talk this way.
    assert framing["adapter"] is None
    assert framing["system"] == prompts.FEEDBACK_SANDWICH_SYSTEM_PROMPT
    # Stage 2 consumes Stage 1's output, behind a prefix carrying the
    # participant's own message — it needs both to write the bread.
    assert framing["user"] is None
    assert USER_TEXT in framing["user_prefix"]
    # ...and it must have room to re-emit the whole of Stage 1.
    assert framing["max_tokens"] > plan["stages"][0]["max_tokens"]


def test_the_sandwich_prompt_validates_feelings_not_the_position():
    """The distinction the prompt was rewritten around on 2026-08-07.

    An earlier version asked the bread to validate the user's POSITION, and
    produced openers that told a user their boast was "a sense of control or
    confidence" immediately before the body called it grade manipulation.
    """
    text = prompts.FEEDBACK_SANDWICH_SYSTEM_PROMPT
    assert "Do NOT validate the POSITION" in text
    assert "Validate the FEELING and the CIRCUMSTANCE" in text
    # The rules that fixed the two structural failures; see the note above the
    # prompt for what each one was worth on the 10-row development sample.
    assert "LENGTH IS NOT YOURS TO CHANGE" in text
    assert "Both are required" in text
    # It must never license weakening the thing the study is measuring.
    assert "Do NOT soften" in text


# --- The shared system prompt ---------------------------------------------

def test_shared_prompt_is_the_ethics_section_only():
    base = prompts.BASE_SYSTEM_PROMPT
    assert "[ETHICS]" in base
    assert "Never request personally identifying information." in base
    assert "If a sensitive topic arises, offer to skip it." in base
    for section in REMOVED_SECTIONS:
        assert section not in base, "shared prompt still contains %s" % section
    assert "[SYCOPHANCY POLICY]" not in base


def test_every_arm_receives_the_shared_system_prompt():
    """The system prompt must no longer be confounded with the arm."""
    for arm in prompts.ARMS:
        plan = prompts.build_generation_plan(arm, USER_TEXT)
        system = plan["stages"][0]["system"]
        assert system is not None, "arm %s has no system prompt" % arm
        assert system.startswith(prompts.BASE_SYSTEM_PROMPT), (
            "arm %s does not start from the shared system prompt" % arm
        )


def test_removed_sections_are_absent_from_every_stage_of_every_arm():
    for arm in prompts.ARMS:
        plan = prompts.build_generation_plan(arm, USER_TEXT)
        for stage in plan["stages"]:
            if stage["system"] is None:
                continue
            for section in REMOVED_SECTIONS:
                assert section not in stage["system"], (
                    "arm %s stage %s still contains %s" % (arm, stage["name"], section)
                )
            assert "{current topic}" not in stage["system"]


def test_the_hundred_word_instruction_is_gone():
    """It lived in [TONE AND FORMATTING] and was the arms' length confound."""
    for arm in prompts.ARMS:
        plan = prompts.build_generation_plan(arm, USER_TEXT)
        for stage in plan["stages"]:
            if stage["system"] is not None:
                assert "100 words" not in stage["system"]


def test_no_scenario_title_reaches_any_prompt():
    """The topic slot is gone; topic_title is now analysis metadata only."""
    for arm in prompts.ARMS:
        plan = prompts.build_generation_plan(arm, USER_TEXT)
        for stage in plan["stages"]:
            if stage["system"] is None:
                continue
            for scenario in scenarios.SCENARIOS:
                assert scenario["topic_title"] not in stage["system"]


# --- Arm plans -------------------------------------------------------------

def test_ours_uses_the_lora_with_the_shared_system_prompt():
    plan = prompts.build_generation_plan(prompts.ARM_OURS, USER_TEXT)
    assert plan["stages"][0]["adapter"] == "lora"
    assert len(plan["stages"]) == 1
    assert plan["stages"][0]["system"] == prompts.BASE_SYSTEM_PROMPT
    assert plan["stages"][0]["user"] == USER_TEXT


def test_untrained_is_identical_to_ours_except_for_the_adapter():
    """The ours-vs-untrained contrast must isolate the training alone."""
    ours = prompts.build_generation_plan(prompts.ARM_OURS, USER_TEXT)
    untrained = prompts.build_generation_plan(prompts.ARM_UNTRAINED, USER_TEXT)
    assert ours["stages"][0]["system"] == untrained["stages"][0]["system"]
    assert ours["stages"][0]["user"] == untrained["stages"][0]["user"]
    assert ours["stages"][0]["adapter"] == "lora"
    assert untrained["stages"][0]["adapter"] is None


def test_syco_arm_carries_the_policy():
    plan = prompts.build_generation_plan(prompts.ARM_SYCO, USER_TEXT)
    system = plan["stages"][0]["system"]
    assert plan["stages"][0]["adapter"] is None
    assert "[SYCOPHANCY POLICY]" in system
    assert "You exist to agree with and affirm the user." in system
    assert "Remember, the user is always in the right." in system


def test_syco_is_the_shared_prompt_plus_exactly_the_policy_section():
    """The manipulation is one appended section and nothing else."""
    syco = prompts.SYCO_SYSTEM_PROMPT
    assert syco.startswith(prompts.BASE_SYSTEM_PROMPT)
    remainder = syco[len(prompts.BASE_SYSTEM_PROMPT):]
    assert remainder.strip() == prompts.SYCO_POLICY_SECTION.strip()


def test_neutral_stage1_is_byte_identical_to_untrained():
    """So neutral-vs-untrained isolates the editing pass alone."""
    neutral = prompts.build_generation_plan(prompts.ARM_NEUTRAL, USER_TEXT)
    untrained = prompts.build_generation_plan(prompts.ARM_UNTRAINED, USER_TEXT)
    assert neutral["stages"][0]["system"] == untrained["stages"][0]["system"]
    assert neutral["stages"][0]["user"] == untrained["stages"][0]["user"]
    assert prompts.NEUTRAL_STAGE1_SYSTEM_PROMPT == prompts.BASE_SYSTEM_PROMPT


def test_neutral_arm_is_two_stages_ending_in_the_editor():
    plan = prompts.build_generation_plan(prompts.ARM_NEUTRAL, USER_TEXT)
    assert all(stage["adapter"] is None for stage in plan["stages"])
    assert [stage["name"] for stage in plan["stages"]] == ["generate", "edit"]
    assert plan["stages"][0]["user"] == USER_TEXT
    # Stage 2 consumes Stage 1's output, signalled by user=None.
    assert plan["stages"][1]["user"] is None
    assert "rewrites AI assistant messages to be strictly" in plan["stages"][1]["system"]


def test_neutral_stage1_has_no_policy():
    stage1 = prompts.NEUTRAL_STAGE1_SYSTEM_PROMPT
    assert "[SYCOPHANCY POLICY]" not in stage1
    assert "You exist to agree with and affirm the user." not in stage1
    assert "the user is always in the right" not in stage1


def test_neutral_stage1_is_strictly_shorter_than_the_syco_prompt():
    assert len(prompts.NEUTRAL_STAGE1_SYSTEM_PROMPT) < len(prompts.SYCO_SYSTEM_PROMPT)


# --- The composed user turn ------------------------------------------------

def test_user_turn_is_their_description_then_their_perspective():
    turn = prompts.build_user_turn(OWN_DESCRIPTION, PERSPECTIVE)
    assert turn == OWN_DESCRIPTION + "\n\n" + PERSPECTIVE
    assert turn.index(OWN_DESCRIPTION) < turn.index(PERSPECTIVE)


def test_user_turn_strips_surrounding_whitespace_but_keeps_inner_text():
    turn = prompts.build_user_turn("  " + OWN_DESCRIPTION + "\n\n",
                                   "\n " + PERSPECTIVE + "  \n")
    assert turn == OWN_DESCRIPTION + "\n\n" + PERSPECTIVE
    assert not turn.startswith(" ")
    assert not turn.endswith("\n")


def test_user_turn_carries_no_labels_or_scaffolding():
    """Plain first-person prose, matching the OEQ training distribution."""
    turn = prompts.build_user_turn(OWN_DESCRIPTION, PERSPECTIVE)
    for marker in ("Situation:", "Scenario:", "My perspective:", "###", "[",
                   "User:", "Context:"):
        assert marker not in turn


def test_user_turn_rejects_empty_parts():
    for bad in ("", "   ", None):
        with pytest.raises(AssertionError):
            prompts.build_user_turn(bad, PERSPECTIVE)
        with pytest.raises(AssertionError):
            prompts.build_user_turn(OWN_DESCRIPTION, bad)


def test_no_scenario_text_reaches_the_user_turn():
    """REGRESSION GUARD. The vignette is a canned example, not the participant's
    incident. Prepending it made the model answer about the wrong story — e.g.
    lecturing someone about an ex's art show they never mentioned."""
    turn = prompts.build_user_turn(OWN_DESCRIPTION, PERSPECTIVE)
    for scenario in scenarios.SCENARIOS:
        assert scenario["vignette"] not in turn
        assert scenario["title"] not in turn
        assert scenario["topic_title"] not in turn
        assert scenario["question"] not in turn


def test_both_halves_reach_the_model_for_every_arm():
    """The whole point of the concatenation: neither half is dropped."""
    for arm in prompts.ARMS:
        plan = prompts.build_generation_plan(arm, USER_TEXT)
        first = plan["stages"][0]["user"]
        assert OWN_DESCRIPTION in first
        assert PERSPECTIVE in first


def test_participant_text_never_reaches_a_system_prompt():
    """The system prompt must not be a channel for participant content."""
    secret = "MYUNIQUEPARTICIPANTSENTENCE"
    for arm in prompts.ARMS:
        plan = prompts.build_generation_plan(
            arm, prompts.build_user_turn(OWN_DESCRIPTION, secret))
        for stage in plan["stages"]:
            if stage["system"] is not None:
                assert secret not in stage["system"]


# --- Guards ----------------------------------------------------------------

def test_unknown_arm_and_empty_user_text_are_rejected():
    with pytest.raises(AssertionError):
        prompts.build_generation_plan("not_an_arm", USER_TEXT)
    for bad in ("", "   ", None):
        with pytest.raises(AssertionError):
            prompts.build_generation_plan(prompts.ARM_OURS, bad)


def test_arms_tuple_matches_the_declared_constants():
    assert set(prompts.ARMS) == {prompts.ARM_OURS, prompts.ARM_UNTRAINED,
                                 prompts.ARM_SYCO, prompts.ARM_NEUTRAL,
                                 prompts.ARM_OURS_NOREVIEW,
                                 prompts.ARM_UNTRAINED_NOREVIEW,
                                 prompts.ARM_OURS_SANDWICH}
    assert len(prompts.ARMS) == 7
    assert len(set(prompts.ARMS)) == 7, "an arm name is duplicated"


# --- The no-review arms ----------------------------------------------------
# These manipulate the participant's FLOW, not the model. Everything below
# exists to keep that true: the moment a no-review arm's generation diverges
# from its twin's, the pair stops isolating the review step.

@pytest.mark.parametrize("noreview,twin", [
    (prompts.ARM_OURS_NOREVIEW, prompts.ARM_OURS),
    (prompts.ARM_UNTRAINED_NOREVIEW, prompts.ARM_UNTRAINED),
])
def test_a_noreview_arm_generates_exactly_like_its_twin(noreview, twin):
    plan = prompts.build_generation_plan(noreview, USER_TEXT)
    twin_plan = prompts.build_generation_plan(twin, USER_TEXT)
    assert plan["stages"] == twin_plan["stages"]
    # ...and it still reports itself, or the worker's stage record and the
    # exported `arm` column would attribute it to the wrong condition.
    assert plan["arm"] == noreview


def test_ours_noreview_addresses_the_lora_and_untrained_noreview_does_not():
    """The half of the pair that is NOT flow: one still gets the adapter."""
    assert prompts.build_generation_plan(
        prompts.ARM_OURS_NOREVIEW, USER_TEXT)["stages"][0]["adapter"] == "lora"
    assert prompts.build_generation_plan(
        prompts.ARM_UNTRAINED_NOREVIEW, USER_TEXT)["stages"][0]["adapter"] is None


def test_exactly_the_two_noreview_arms_skip_the_review_page():
    assert set(prompts.SKIP_REVIEW_ARMS) == {prompts.ARM_OURS_NOREVIEW,
                                             prompts.ARM_UNTRAINED_NOREVIEW}
    for arm in prompts.ARMS:
        assert prompts.shows_review(arm) is (arm not in prompts.SKIP_REVIEW_ARMS)
    assert prompts.shows_review(prompts.ARM_OURS) is True
    assert prompts.shows_review(prompts.ARM_OURS_NOREVIEW) is False


def test_shows_review_rejects_an_unknown_arm():
    """No silent default: a new arm must declare which flow it walks."""
    with pytest.raises(AssertionError):
        prompts.shows_review("not_an_arm")


def test_generation_arms_drop_the_flow_only_duplicates():
    """What the smoke test walks: every distinct generation, each once."""
    assert set(prompts.GENERATION_ARMS) == set(prompts.ARMS) - set(prompts.SKIP_REVIEW_ARMS)
    # Distinct in what is actually sent: the model addressed AND the stages.
    # `ours` and `untrained` share their stages and differ only in the adapter.
    plans = []
    for arm in prompts.GENERATION_ARMS:
        plan = prompts.build_generation_plan(arm, USER_TEXT)
        signature = plan["stages"]
        assert signature not in plans, "%s duplicates another generation arm" % (arm,)
        plans.append(signature)


# --- Manual-entry conditions (the /debug_<key> links) ----------------------

def test_every_drawable_arm_has_a_manual_entry_condition():
    """The invariant the links exist for.

    Random assignment is what stops you looking at a condition on purpose. The
    links cover exactly the drawable arms, so a link does not reach a flow you
    could not otherwise see — it tells you WHICH MODEL is answering, which the
    draw hides. So the keys must cover exactly RANDOMIZED_ARMS: a drawable arm
    nobody can force is a condition nobody can check by hand.

    c1/c2 still mean "ours"/"untrained"; what changed on 2026-08-07 is that they
    name the review-page arms rather than their no-review twins.
    """
    assert set(prompts.DEBUG_CONDITIONS.values()) == set(prompts.RANDOMIZED_ARMS)
    assert prompts.DEBUG_CONDITIONS["c1"] == prompts.ARM_OURS
    assert prompts.DEBUG_CONDITIONS["c2"] == prompts.ARM_UNTRAINED


def test_every_manual_entry_condition_names_a_real_arm():
    """A key that mapped to nothing would 'work' and force no condition."""
    for key, arm in prompts.DEBUG_CONDITIONS.items():
        assert arm in prompts.ARMS, "condition %r maps to unknown arm %r" % (key, arm)


def test_manual_entry_keys_are_url_and_path_safe():
    """Each key becomes a URL segment (/debug_<key>) and a directory name under
    static/, so anything else would produce a link that cannot be served."""
    for key in prompts.DEBUG_CONDITIONS:
        assert key and key.replace("_", "").isalnum() and key.lower() == key, key


def test_no_two_conditions_point_at_the_same_arm():
    """Two links to one condition is a mislabelled link waiting to happen."""
    arms = list(prompts.DEBUG_CONDITIONS.values())
    assert len(set(arms)) == len(arms), arms
