"""Tests for analysis/export_responses.py.

The export now reads submission rows (from Supabase or the fallback JSONL)
rather than the SQLite database. The properties that matter are unchanged from
the old exporter and are the ones asserted here: the column set comes from
questionnaires.py rather than from the data, nothing is imputed, and the QSF's
skipped choice ids are translated to scale positions correctly.
"""

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))

import export_responses
import questionnaires
import scenarios
from flow_helpers import all_answers


def make_submission(session_id="s1", **overrides):
    order = scenarios.scenario_keys()
    row = {
        "session_id": session_id,
        "arm": "ours",
        # Present and false on a real row: the arm was drawn, not forced by a
        # /debug_<key> link.
        "is_debug": False,
        "job_id": "j1",
        "topic_title": "Excluding Someone",
        "user_text": "their story\n\ntheir side",
        "shown_response": "the model's reply",
        "shown_response_chars": len("the model's reply"),
        "stages": [{"name": "generate", "output": "stage one"},
                   {"name": "edit", "output": "stage two"}],
        "worker_attempts": 1,
        "manual_retries": 0,
        "prolific_pid": "PID1",
        "consent": "agree",
        "eligibility": "eligible",
        "selected_scenario": order[0],
        "screening_order": order,
        "screening": [{"scenario_key": order[0], "position": 0, "similarity": "very",
                       "description": "what happened"}],
        "perspective_text": "my side",
        "additional_thoughts": "thanks",
        "answers": all_answers(),
        "missing_keys": [],
        "client_timings": {},
        "session_created_at": "2026-07-28T00:00:00.000000Z",
        "submitted_at": "2026-07-28T00:10:00.000000Z",
    }
    row.update(overrides)
    return row


# --- Column set ------------------------------------------------------------

def test_every_questionnaire_key_gets_a_column_even_when_absent_from_the_data():
    """The schema comes from questionnaires.py, not from whatever arrived."""
    columns, _ = export_responses.build_participant_rows(
        [make_submission(answers={})])
    for key in questionnaires.all_response_keys():
        assert key in columns


def test_unanswered_items_are_empty_cells_and_are_never_imputed():
    answers = all_answers()
    del answers["fk_1"]
    columns, rows = export_responses.build_participant_rows(
        [make_submission(answers=answers, missing_keys=["fk_1"])])
    assert rows[0]["fk_1"] == ""
    assert rows[0]["fk_1_point"] == ""
    assert rows[0]["n_missing_answers"] == 1


def test_a_parallel_point_column_exists_for_every_scaled_key():
    columns, _ = export_responses.build_participant_rows([make_submission()])
    for question in questionnaires.all_questions():
        if not any("scale_point" in option for option in question["options"]):
            continue
        for key in questionnaires.response_keys(question):
            assert "%s_point" % key in columns


def test_skipped_qsf_choice_ids_map_to_their_display_position():
    """response_objectivity stores 8 for scale point 7; both must be exported."""
    answers = all_answers()
    answers["response_objectivity"] = 8
    _, rows = export_responses.build_participant_rows([make_submission(answers=answers)])
    assert rows[0]["response_objectivity"] == 8
    assert rows[0]["response_objectivity_point"] == 7


# --- Derived columns -------------------------------------------------------

def test_attention_checks_are_scored_and_a_skip_is_distinguishable_from_a_failure():
    passing = all_answers()
    for key in ("attention_check_1", "attention_check_2", "attention_check_3"):
        passing[key] = 1
    passing["AIAS_5"] = 10
    _, rows = export_responses.build_participant_rows([make_submission(answers=passing)])
    assert rows[0]["attention_check_passed"] is True
    assert rows[0]["aias_attention_passed"] is True

    failing = dict(passing)
    failing["attention_check_2"] = 4
    failing["AIAS_5"] = 3
    _, rows = export_responses.build_participant_rows([make_submission(answers=failing)])
    assert rows[0]["attention_check_passed"] is False
    assert rows[0]["aias_attention_passed"] is False

    _, rows = export_responses.build_participant_rows([make_submission(answers={})])
    assert rows[0]["attention_check_passed"] is None
    assert rows[0]["aias_attention_passed"] is None


def test_duration_is_computed_from_the_two_timestamps():
    _, rows = export_responses.build_participant_rows([make_submission()])
    assert rows[0]["duration_seconds"] == 600.0


def test_timestamps_from_supabase_parse_as_well_as_the_servers_own_form():
    row = make_submission(session_created_at="2026-07-28T00:00:00+00:00",
                          submitted_at="2026-07-28T00:05:00+00:00")
    _, rows = export_responses.build_participant_rows([row])
    assert rows[0]["duration_seconds"] == 300.0


# --- Screening and generations --------------------------------------------

def test_screening_rows_mark_the_selected_scenario():
    order = scenarios.scenario_keys()
    row = make_submission(screening=[
        {"scenario_key": order[0], "position": 0, "similarity": "not_really",
         "description": None},
        {"scenario_key": order[1], "position": 1, "similarity": "very",
         "description": "the real one"},
    ], selected_scenario=order[1])
    _, rows = export_responses.build_screening_rows([row])
    assert [r["selected"] for r in rows] == [False, True]
    assert rows[0]["description"] == "" and rows[0]["description_chars"] == 0
    assert rows[1]["description_chars"] == len("the real one")


def test_generation_rows_break_out_both_stages():
    _, rows = export_responses.build_generation_rows([make_submission()])
    assert rows[0]["stage1_output"] == "stage one"
    assert rows[0]["stage2_output"] == "stage two"
    assert rows[0]["n_stages"] == 2
    assert rows[0]["user_text"] == "their story\n\ntheir side"


def test_a_submission_with_no_generation_contributes_no_generation_row():
    """Screen-outs and decliners never reached the model."""
    _, rows = export_responses.build_generation_rows(
        [make_submission(job_id=None, eligibility="screened_out")])
    assert rows == []


# --- Sources ---------------------------------------------------------------

def test_reading_the_fallback_jsonl_keeps_the_last_row_per_session(tmp_path):
    path = tmp_path / "submissions.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(make_submission("s1", prolific_pid="first")) + "\n")
        handle.write(json.dumps(make_submission("s2")) + "\n")
        handle.write(json.dumps(make_submission("s1", prolific_pid="second")) + "\n")

    rows = export_responses.read_local(str(path))
    assert len(rows) == 2
    by_id = {row["session_id"]: row for row in rows}
    assert by_id["s1"]["prolific_pid"] == "second"


def test_reading_a_missing_fallback_file_fails_loudly(tmp_path):
    with pytest.raises(AssertionError):
        export_responses.read_local(str(tmp_path / "nope.jsonl"))


# --- Writing ---------------------------------------------------------------

def test_the_written_csv_round_trips(tmp_path):
    columns, rows = export_responses.build_participant_rows([make_submission()])
    path = tmp_path / "participants.csv"
    written = export_responses.write_csv(str(path), columns, rows)
    assert written == 1

    with open(path, newline="", encoding="utf-8") as handle:
        read_back = list(csv.DictReader(handle))
    assert len(read_back) == 1
    assert read_back[0]["session_id"] == "s1"
    assert read_back[0]["arm"] == "ours"
    assert list(read_back[0]) == columns


def test_write_csv_refuses_a_row_carrying_an_undeclared_column(tmp_path):
    with pytest.raises(ValueError):
        export_responses.write_csv(str(tmp_path / "x.csv"), ["a"], [{"a": 1, "b": 2}])


# --- Hand-walked sessions --------------------------------------------------

def test_forced_arm_sessions_are_dropped_by_default():
    """A /debug_<key> session chose its own arm, so it is not a participant.

    This filter is what makes the debug links safe to leave unsecured: whatever
    gets walked through them is quarantined rather than counted in a cell.
    """
    rows = [make_submission("real", is_debug=False),
            make_submission("hand", is_debug=True, arm="ours_noreview")]
    kept, dropped = export_responses.drop_debug_sessions(rows)
    assert [row["session_id"] for row in kept] == ["real"]
    assert dropped == 1


def test_rows_without_the_flag_are_kept():
    """The fallback JSONL predates the column, and those rows are participants."""
    rows = [make_submission("old")]
    del rows[0]["is_debug"]
    kept, dropped = export_responses.drop_debug_sessions(rows)
    assert len(kept) == 1 and dropped == 0


def test_the_flag_is_exported_as_a_column():
    """Kept or dropped, an analysis must be able to see which rows were forced."""
    columns, rows = export_responses.build_participant_rows(
        [make_submission("hand", is_debug=True)])
    assert "is_debug" in columns
    assert rows[0]["is_debug"] is True
