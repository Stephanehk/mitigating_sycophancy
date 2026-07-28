"""Tests for flow.py: randomization and whole-payload submission validation.

The stage machine that used to be tested here lives in the browser now
(static/app.js), so what remains are the two things the server still refuses to
delegate — the randomization draws — and the validation of the single payload it
accepts at the end.
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import flow
import questionnaires
import scenarios
from prompts import ARMS


# --- Randomization ---------------------------------------------------------

def test_screening_order_is_a_permutation_of_every_scenario():
    for seed in range(20):
        order = flow.new_screening_order(random.Random(seed))
        assert sorted(order) == sorted(scenarios.scenario_keys())


def test_screening_order_actually_varies():
    orders = {tuple(flow.new_screening_order(random.Random(seed))) for seed in range(30)}
    assert len(orders) > 1, "screening order never changed across 30 seeds"


def test_assign_arm_only_ever_returns_a_declared_arm():
    for seed in range(50):
        assert flow.assign_arm(random.Random(seed)) in ARMS


def test_every_arm_is_reachable():
    drawn = {flow.assign_arm(random.Random(seed)) for seed in range(200)}
    assert drawn == set(ARMS)


# --- Free text -------------------------------------------------------------

def test_clean_text_enforces_the_minimum_length():
    with pytest.raises(flow.ValidationError):
        flow.clean_text("too short", "perspective", min_chars=50)


def test_clean_text_rejects_blank_when_required_and_allows_it_otherwise():
    with pytest.raises(flow.ValidationError):
        flow.clean_text("   ", "perspective", min_chars=0, required=True)
    assert flow.clean_text("   ", "feedback", min_chars=0, required=False) is None


def test_clean_text_rejects_text_over_the_cap():
    with pytest.raises(flow.ValidationError):
        flow.clean_text("x" * (config.MAX_TEXT_CHARS + 1), "perspective", min_chars=1)


def test_validate_model_input_returns_the_scenario_and_both_texts():
    key = scenarios.scenario_keys()[0]
    scenario, description, perspective = flow.validate_model_input(
        key, "d" * 60, "p" * 100)
    assert scenario["key"] == key
    assert description == "d" * 60
    assert perspective == "p" * 100


def test_validate_model_input_rejects_an_unknown_scenario():
    with pytest.raises(flow.ValidationError):
        flow.validate_model_input("not_a_scenario", "d" * 60, "p" * 100)


def test_validate_model_input_enforces_both_minimum_lengths():
    key = scenarios.scenario_keys()[0]
    with pytest.raises(flow.ValidationError):
        flow.validate_model_input(key, "short", "p" * 100)
    with pytest.raises(flow.ValidationError):
        flow.validate_model_input(key, "d" * 60, "short")


# --- Questionnaire answers -------------------------------------------------

def _every_answer(value_index=0):
    answers = {}
    for question in questionnaires.all_questions():
        option = question["options"][min(value_index, len(question["options"]) - 1)]
        for key in questionnaires.response_keys(question):
            answers[key] = option["value"]
    return answers


def test_a_complete_answer_set_validates_with_nothing_missing():
    normalized, missing = flow.validate_questionnaire_answers(_every_answer())
    assert missing == []
    assert set(normalized) == set(questionnaires.all_response_keys())


def test_missing_answers_are_reported_not_raised():
    """A blank cell must not throw away an otherwise complete session.

    Every question is required and the client blocks Continue, so a gap here is
    a client bug — but this validation runs at the very end of the study, and
    rejecting would lose the whole participant over one cell.
    """
    answers = _every_answer()
    dropped = questionnaires.all_response_keys()[0]
    del answers[dropped]
    normalized, missing = flow.validate_questionnaire_answers(answers)
    assert missing == [dropped]
    assert dropped not in normalized


def test_out_of_range_codes_are_rejected():
    answers = _every_answer()
    answers[questionnaires.all_response_keys()[0]] = 999
    with pytest.raises(flow.ValidationError):
        flow.validate_questionnaire_answers(answers)


def test_skipped_qsf_choice_ids_are_preserved_and_enforced():
    """response_objectivity's options are 1,2,3,4,5,7,8 — 6 is not a valid code."""
    question = [q for q in questionnaires.all_questions()
                if q["id"] == "response_objectivity"][0]
    values = [option["value"] for option in question["options"]]
    assert values == [1, 2, 3, 4, 5, 7, 8]

    answers = _every_answer()
    answers["response_objectivity"] = 8
    normalized, _ = flow.validate_questionnaire_answers(answers)
    assert normalized["response_objectivity"] == 8

    answers["response_objectivity"] = 6
    with pytest.raises(flow.ValidationError):
        flow.validate_questionnaire_answers(answers)


def test_numeric_strings_are_accepted_but_booleans_and_junk_are_not():
    answers = _every_answer()
    answers["fk_1"] = "5"
    normalized, _ = flow.validate_questionnaire_answers(answers)
    assert normalized["fk_1"] == 5

    for junk in (True, "five", [], {}):
        answers["fk_1"] = junk
        with pytest.raises(flow.ValidationError):
            flow.validate_questionnaire_answers(answers)


def test_unknown_answer_keys_are_dropped_rather_than_stored():
    answers = _every_answer()
    answers["not_a_real_question"] = 3
    normalized, _ = flow.validate_questionnaire_answers(answers)
    assert "not_a_real_question" not in normalized


def test_malformed_answers_block_is_rejected():
    with pytest.raises(flow.ValidationError):
        flow.validate_questionnaire_answers(["not", "a", "dict"])


# --- Screening answers -----------------------------------------------------

def test_screening_answers_recover_their_presentation_position():
    order = ["exclusion", "relationship", "others", "uncomfortable"]
    result = flow.validate_screening_answers(
        [{"scenario_key": "others", "similarity": "very", "description": "d" * 30}], order)
    assert result[0]["position"] == 2


def test_screening_description_is_dropped_when_the_answer_did_not_require_one():
    order = scenarios.scenario_keys()
    result = flow.validate_screening_answers(
        [{"scenario_key": order[0], "similarity": "not_really",
          "description": "junk that should not be kept"}], order)
    assert result[0]["description"] is None


def test_screening_rejects_unknown_scenarios_and_similarity_values():
    order = scenarios.scenario_keys()
    with pytest.raises(flow.ValidationError):
        flow.validate_screening_answers(
            [{"scenario_key": "nope", "similarity": "very", "description": "d" * 30}], order)
    with pytest.raises(flow.ValidationError):
        flow.validate_screening_answers(
            [{"scenario_key": order[0], "similarity": "maybe"}], order)


def test_screening_rejects_a_duplicated_scenario():
    order = scenarios.scenario_keys()
    entry = {"scenario_key": order[0], "similarity": "not_really"}
    with pytest.raises(flow.ValidationError):
        flow.validate_screening_answers([entry, dict(entry)], order)


# --- Whole payload ---------------------------------------------------------

def _payload(**overrides):
    order = scenarios.scenario_keys()
    payload = {
        "consent": "agree",
        "prolific_pid": "PID",
        "eligibility": "eligible",
        "selected_scenario": order[0],
        "screening": [{"scenario_key": order[0], "similarity": "very",
                       "description": "d" * 60}],
        "perspective_text": "p" * 100,
        "additional_thoughts": "",
        "answers": _every_answer(),
        "client_timings": {"consent": "2026-07-28T00:00:00Z"},
    }
    payload.update(overrides)
    return payload


def test_a_well_formed_payload_normalizes():
    clean = flow.validate_submission(_payload(), scenarios.scenario_keys())
    assert clean["consent"] == "agree"
    assert clean["eligibility"] == "eligible"
    assert clean["missing_keys"] == []
    assert clean["additional_thoughts"] is None


def test_payload_requires_consent_and_eligibility():
    for field in ("consent", "eligibility"):
        with pytest.raises(flow.ValidationError):
            flow.validate_submission(_payload(**{field: None}), scenarios.scenario_keys())
        with pytest.raises(flow.ValidationError):
            flow.validate_submission(_payload(**{field: "nonsense"}), scenarios.scenario_keys())


def test_an_eligible_payload_must_name_a_selected_scenario():
    with pytest.raises(flow.ValidationError):
        flow.validate_submission(_payload(selected_scenario=None), scenarios.scenario_keys())


def test_a_screened_out_payload_needs_no_selected_scenario():
    clean = flow.validate_submission(
        _payload(eligibility="screened_out", selected_scenario=None),
        scenarios.scenario_keys())
    assert clean["selected_scenario"] is None


def test_payload_rejects_an_unknown_selected_scenario():
    with pytest.raises(flow.ValidationError):
        flow.validate_submission(_payload(selected_scenario="nope"),
                                 scenarios.scenario_keys())


def test_free_text_is_capped_at_the_configured_limit():
    clean = flow.validate_submission(
        _payload(perspective_text="p" * (config.MAX_TEXT_CHARS + 500),
                 additional_thoughts="a" * (config.MAX_TEXT_CHARS + 500)),
        scenarios.scenario_keys())
    assert len(clean["perspective_text"]) == config.MAX_TEXT_CHARS
    assert len(clean["additional_thoughts"]) == config.MAX_TEXT_CHARS


def test_prolific_id_is_trimmed_and_capped():
    clean = flow.validate_submission(_payload(prolific_pid="  " + "x" * 500 + "  "),
                                     scenarios.scenario_keys())
    assert len(clean["prolific_pid"]) == 200


def test_malformed_payload_and_timings_are_rejected():
    with pytest.raises(flow.ValidationError):
        flow.validate_submission("not a dict", scenarios.scenario_keys())
    with pytest.raises(flow.ValidationError):
        flow.validate_submission(_payload(client_timings=["nope"]),
                                 scenarios.scenario_keys())


# --- Supabase URL normalization --------------------------------------------

def test_the_dashboards_api_url_is_trimmed_to_the_project_origin():
    """The dashboard shows `.../rest/v1/` next to the API keys people copy.

    supabase_client appends `/rest/v1/<table>` itself, so the untrimmed form
    would 404 at submit time — after a participant has finished.
    """
    origin = "https://examplerefabcdefgh.supabase.co"
    for given in (origin,
                  origin + "/",
                  origin + "/rest/v1",
                  origin + "/rest/v1/"):
        assert config._normalize_supabase_url(given) == origin, given


def test_normalization_leaves_a_placeholder_alone_for_the_boot_check():
    placeholder = "https://<project-ref>.supabase.co"
    assert config._normalize_supabase_url(placeholder) == placeholder
